"""
End-to-end pipeline for learning concentration from morphology and rheology
using Gaussian Process Regression.

### Purpose:
This script trains a **Gaussian Process (GP)** to predict concentration from morphological and rheological features.
It includes data loading, preprocessing, hyperparameter optimization (Bayesian Optimization + grid search),
and evaluation of the best model.

### Pipeline Stages:
1. **Load Data**: Morphology, rheology, and concentration data are loaded and aligned.
2. **Data Cleaning**: Invalid or missing entries are filtered out.
3. **Data Augmentation**: Synthetic data is generated using measurement uncertainty.
4. **Feature Preprocessing**: Features are scaled and weighted.
5. **Hyperparameter Search**: Bayesian Optimization (BO) and grid search are used to find optimal feature weights.
6. **Model Training**: The best GP is trained and evaluated.
7. **Export Results**: Predictions, landscapes, and evaluation metrics are saved.

### Key Files:
- **Input**: Morphology data (in `Data/`), rheology/concentration (Excel file).
- **Output**:
  - `predictions_diameter_fixed.csv`: Test set predictions and uncertainties.
  - `parameter_ablation.csv`: Ablation study results.
  - `combined_weight_landscape_diameter_fixed.csv`: Grid search results.
  - `grid_stage1_coarse.csv`: Coarse grid search results.

### Configuration:
- **AUGMENT**: Enable/disable synthetic data generation.
- **SCALE_TARGET**: Standardize target values (concentration).
- **FIXED_DIAMETER**: Fix diameter weight to remove scale invariance.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import time

from functools import lru_cache
from tqdm import tqdm

from MatrixGatherer import MorphologyDataLoader  # Loads morphology/rheology/concentration data
from DataGeneration import DataGeneration        # Generates synthetic data from uncertainty
from DataProcessing import Processor            # Preprocesses features (scaling, weighting)

# --- Scikit-learn ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


# --- Bayesian Optimization ---
from skopt import gp_minimize                     # Bayesian optimization
from skopt.space import Real                     # Defines search space
from skopt.utils import use_named_args            # Named arguments for BO
from itertools import product                   # Grid search combinations



# ----------------------------
# Configuration
# ----------------------------
# --- Paths ---
CURRENT_DIR = Path(__file__).parent
BASE_PATH = CURRENT_DIR.parent / "Data"                  # Folder for morphology data
EXCEL_FILE = CURRENT_DIR.parent / "Master Sheet - Change.xlsx"  # Rheology/concentration Excel
LANDSCAPE_FILE = CURRENT_DIR / "combined_weight_landscape_diameter_fixed.csv"  # Grid search results

# --- Features ---
ACTIVE_DESCRIPTORS = ["K", "n"]  # Features to use
INACTIVE_DESCRIPTORS = ["diameter", "circularity", "compactness"]  # Unused features (if any)

# --- Experimental Switches ---
AUGMENT = True          # Generate synthetic data from mean ± std
SCALE_TARGET = False    # Standardize target (concentration) values

# --- Reproducibility ---
RANDOM_STATE = 42       # Random seed for reproducibility

# --- Gauge Fixing ---
FIXED_DIAMETER = 1.0    # Fix diameter weight to remove scale invariance between feature weights



# ----------------------------
# Modeling helpers
# ----------------------------


def build_gp(random_state=42):
    """
    Constructs a Gaussian Process Regressor with:
    - **RBF kernel**: For smooth nonlinear relationships.
    - **ConstantKernel**: Scales the covariance.
    - **WhiteKernel**: Models noise.
    The kernel is defined as:
    \[ k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{||x_i - x_j||^2}{2l^2}\right) + \sigma_n^2 \delta_{ij} \]
    where:
    - \(\sigma_f^2\) is the signal variance (ConstantKernel).
    - \(l\) is the length scale (RBF).
    - \(\sigma_n^2\) is the noise variance (WhiteKernel).
    """
    kernel = ConstantKernel(1.0, (1e-10, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e7)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e2))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,  # Number of kernel optimizer restarts
        random_state=random_state,
        normalize_y=False         # Do not standardize target values
    )
    return gp

def evaluate_weights(weights, X_vf, X_sel_rheo, y, process):
    """
    Evaluates a given set of feature weights using cross-validated R².
    Steps:
    1. Apply weights to morphology and rheology features.
    2. Concatenate weighted feature blocks.
    3. Train and evaluate a Gaussian Process via 3-fold CV.

    Returns:
        Mean cross-validated R² score.
    """
    # --- Apply weights to morphology features ---
    W_morph = {k: weights[k] for k in ['diameter', 'circularity', 'compactness']}
    X_m = process.apply_manual_weights(X_vf, W_morph)

    # --- Apply weights to rheology features ---
    W_rheo = {k: weights[k] for k in ['K', 'n']}
    X_r = process.apply_manual_weights_rheo(X_sel_rheo, W_rheo)

    # --- Combine features ---
    Xc = np.hstack([X_m, X_r])

    # --- Evaluate using 3-fold cross-validation ---
    scores = cross_val_score(
        build_gp(RANDOM_STATE),
        Xc, y,
        cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2"
    )
    return scores.mean()

# ----------------------------
# Grid Search
# ----------------------------
def already_done(weights, df_prev, tol=1e-6):
    """
    Checks if a weight combination was already evaluated in previous runs.
    - **weights**: Dictionary of feature weights.
    - **df_prev**: DataFrame of previous evaluations.
    - **tol**: Tolerance for numerical comparison.
    Returns True if the weights were already evaluated.
    """
    if len(df_prev) == 0:
        return False
    mask = np.ones(len(df_prev), dtype=bool)
    for k, v in weights.items():
        mask &= np.isclose(df_prev[k], v, atol=tol)
    return mask.any()

def run_grid_search_unified(grid, X_vf, X_sel_rheo, y, process, all_results, df_prev, verbose=True):
    """
    Runs a grid search over feature weights.
    - Skips already evaluated combinations.
    - Saves results to `all_results` and returns best score/weights.
    - **grid**: Dictionary of parameter grids.
    - **all_results**: List to store all evaluation results.
    - **df_prev**: DataFrame of previous evaluations.
    Returns: best_score, best_weights, DataFrame of results.
    """
    start_time = time.time()
    results = []
    best_score = -np.inf
    best_weights = None

    # --- Generate all combinations of grid parameters ---
    keys, values = zip(*grid.items())
    combinations = list(product(*values))
    total = len(combinations)

    print(f"[GRID] Evaluating up to {total} combinations (skipping duplicates)")

    for i, combination in enumerate(tqdm(combinations, desc="Grid search"), 1):
        weights = dict(zip(keys, combination))

        # --- Skip if already evaluated ---
        if already_done(weights, df_prev):
            continue

        # --- Evaluate weights ---
        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process)

        # --- Store results ---
        row = {**weights, **{f"{k}_descriptor": weights[k] for k in weights}, "cv_r2": score}
        results.append(row)
        all_results.append(row)

        # --- Track best score ---
        if score > best_score:
            best_score = score
            best_weights = weights.copy()

        # --- Print progress ---
        if verbose and (i % 10 == 0 or i == 1):
            elapsed = time.time() - start_time
            print(f"[GRID {i}/{total}] elapsed={elapsed:.1f}s, cv_r2={score:.4f}")

    return best_score, best_weights, pd.DataFrame(results)






# ----------------------------
# Main
# ----------------------------
def main():

    # ============================================================
    #    1. LOAD RAW DATA
    # ============================================================
    loader = MorphologyDataLoader(BASE_PATH)
    X_mean, X_std, morph_names = loader.load_with_std()  # Morphology data (mean ± std)
    target_rheo = loader.load_excel_rheology(EXCEL_FILE)  # Rheology data (K, n)
    target_conc = loader.load_excel_concentrations(EXCEL_FILE)  # Concentration data

    print(f"[INFO] Morphology samples: {len(morph_names)}")
    print(f"[INFO] Rheology samples  : {len(target_rheo)}")
    print(f"[INFO] Concentration     : {len(target_conc)}")

    # Index map for morphology
    morph_index = {name: i for i, name in enumerate(morph_names)}

    # ==============================================================  
    #    2. CLEAN RHEOLOGY 
    # ==============================================================  
    clean_rheology = {}
    invalid_rheology = []

    for name, raw in target_rheo.items():
        try:
            mean1 = float(raw[0])
        except Exception:
            mean1 = np.nan
        try:
            mean2 = float(raw[1])
        except Exception:
            mean2 = np.nan
        try:
            sd1 = float(raw[2])
        except Exception:
            sd1 = np.nan
        try:
            sd2 = float(raw[3])
        except Exception:
            sd2 = np.nan

        clean_rheology[name] = (mean1, mean2, sd1, sd2)

        # mark as invalid if mean missing
        if np.isnan(mean1) or np.isnan(mean2):
            invalid_rheology.append(name)


    # ==============================================================  
    #    3. ALIGN SAMPLES
    # ============================================================== 
    # Only keep samples with valid morphology, rheology, and concentration 
    final_samples = [
        name for name in morph_names
        if (name in target_conc) and (name in clean_rheology) and (name not in invalid_rheology)
    ]
    all_results = []


    print(f"[INFO] Final aligned sample count: {len(final_samples)}")

    # ============================================================
    # 4. EXTRACT ALIGNED DATA
    # ============================================================
    X_mean_final = []
    X_std_final = []
    X_mean_rheo_final = []
    X_std_rheo_final = []
    y_final = []
    y_sd_final = []

    for name in final_samples:
        # --- Morphology ---
        idx = morph_index[name]            # correct index look-up
        X_mean_final.append(X_mean[idx])
        X_std_final.append(X_std[idx])

        # --- Rheology (from cleaned dict) ---
        mean1, mean2, sd1, sd2 = clean_rheology[name]
        X_mean_rheo_final.append([mean1, mean2])
        X_std_rheo_final.append([sd1, sd2])

        # --- Concentration ---
        c_mean, c_sd = target_conc[name]
        try:
            c_mean = float(c_mean)
        except Exception:
            c_mean = np.nan

        try:
            c_sd = float(c_sd)
            if c_sd <= 0 or np.isnan(c_sd):
                c_sd = 0.3 * c_mean
        except Exception:
            c_sd = 0.3 * c_mean

        y_final.append(c_mean)
        y_sd_final.append(c_sd)

    # Convert lists to arrays (aligned and same order)
    X_mean = np.array(X_mean_final, dtype=float)
    X_std  = np.array(X_std_final, dtype=float)
    X_mean_rheo = np.array(X_mean_rheo_final, dtype=float)
    X_std_rheo  = np.array(X_std_rheo_final, dtype=float)
    y_mean = np.array(y_final, dtype=float)
    y_sd = np.array(y_sd_final, dtype=float)
    sample_names = final_samples

    # quick sanity prints
    print("[DEBUG] shapes after alignment:",
          "X_mean", X_mean.shape,
          "X_std", X_std.shape,
          "X_mean_rheo", X_mean_rheo.shape,
          "X_std_rheo", X_std_rheo.shape,
          "y", y_mean.shape)





    # ============================================================
    # 5. DATA AUGMENTATION
    # ============================================================
    if AUGMENT:
        augment = DataGeneration()
        X, y, sample_names = augment.generate(X_mean, X_std, y_mean, y_sd, final_samples)
        print(f"[INFO] Augmented dataset size: {len(X)}")
        X_rheo, y, sample_names = augment.generate_rheo(X_mean_rheo, X_std_rheo, y_mean, y_sd, final_samples)
        print(f"[INFO] Augmented dataset size: {len(X_rheo)}")
    else:
        X = X_mean.copy()
        X_rheo = X_mean_rheo.copy()

    # Preprocessing
    process = Processor()
    X_sel = process.fit_transform(X)
    X_vf = process.weight_by_vf_vectorized(X_sel) 
    X_sel_rheo = process.fit_transform_rheo(X_rheo)

    # ============================================================
    # 7. LOAD PREVIOUS RESULTS (IF ANY)
    # ============================================================
    RESULTS_FILE = LANDSCAPE_FILE

    if RESULTS_FILE.exists():
        df_prev = pd.read_csv(RESULTS_FILE)
        print(f"[INFO] Loaded {len(df_prev)} previous evaluations")
    else:
        df_prev = pd.DataFrame()

    
    # ============================================================
    # STAGE 0 — PARAMETER ABLATION
    # ============================================================
    # Purpose:
    #   (A) Measure absolute explanatory power of each descriptor alone
    #   (B) Measure marginal importance via leave-one-out BO
    #
    # Design principles:
    #   - Single-descriptor models are true 1D GPs → no BO needed
    #   - Multi-descriptor models require gauge fixing → diameter = 1
    # ============================================================

    print("\n[ABLATION] Starting parameter ablation study...")

    descriptors = ACTIVE_DESCRIPTORS
    ablation_rows = []

    # ------------------------------------------------------------
    # PART A — Single-descriptor models (pure 1D)
    # ------------------------------------------------------------
    print("[ABLATION A] Single-descriptor models")

    for d in descriptors:
        weights = {k: 0.0 for k in ACTIVE_DESCRIPTORS + INACTIVE_DESCRIPTORS}
        # Note: scale is arbitrary in 1D; gauge fixing is irrelevant here
        weights[d] = 1.0  # arbitrary nonzero scale


        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process)

        ablation_rows.append({
            "mode": "single_descriptor",
            "ablated": None,
            "active": d,
            "cv_r2": score
        })

        print(f"  [1D] {d:12s} → R² = {score:.4f}")


    # ------------------------------------------------------------
    # PART B — Leave-one-out BO (multi-D with gauge fixing)
    # ------------------------------------------------------------
    print("\n[ABLATION B] Leave-one-out BO")

    for ablated in ACTIVE_DESCRIPTORS:

        # Diameter special case
        if ablated == "diameter":
            fixed_diameter = 0.0
        else:
            fixed_diameter = FIXED_DIAMETER

        # Define BO search space for remaining parameters
        space = []
        param_names = []

        for p in ACTIVE_DESCRIPTORS:
            if p == ablated or p == "diameter":
                continue
            space.append(Real(0.0, 5.0, name=p))

        # --- Objective: Maximize R² (minimize negative R²) ---
        @use_named_args(space)
        def loo_objective(**params):
            weights = {k: 0.0 for k in ACTIVE_DESCRIPTORS + INACTIVE_DESCRIPTORS}
            weights.update(params)
            weights[ablated] = 0.0
            weights["diameter"] = 0.0
            weights["circularity"] = 0.0
            weights["compactness"] = 0.0


            score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process)
            return -score
        
        # --- Run BO ---
        result = gp_minimize(
            loo_objective,
            dimensions=space,
            n_calls=30,
            n_initial_points=10,
            random_state=RANDOM_STATE,
        )

        best_score = -result.fun

        ablation_rows.append({
            "mode": "leave_one_out",
            "ablated": ablated,
            "active": "all_except_" + ablated,
            "cv_r2": best_score
        })

        print(f"  [LOO] removed {ablated:12s} → R² = {best_score:.4f}")

    # Save ablation results
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(CURRENT_DIR / "parameter_ablation.csv", index=False)

    print("[ABLATION] Results saved to parameter_ablation.csv")


    # ============================================================
    # STAGE 1 — Bayesian Optimization
    # ============================================================
    # Purpose:
    #   Quickly locate a high-performing region in the 4D weight space
    #   using a surrogate GP and expected improvement acquisition.

    print("Starting the Bayesian Optimizer...")
    space = [
        Real(0, 5.0, name="K"),
        Real(0, 5.0, name="n")
    ]

    @use_named_args(space)
    def objective(K, n):
        weights = {
            "diameter": 0,       
            "circularity": 0, 
            "compactness": 0, 
            "K": K,
            "n": n 
        }

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process)

        all_results.append({**weights, "cv_r2": score})
        return -score



    bo_result = gp_minimize(
        objective,
        dimensions=space,
        n_calls = 15 if len(df_prev) > 50 else 40,
        n_initial_points=10,
        random_state=RANDOM_STATE,
    )
    best_bo_r2 = -bo_result.fun
    print(f"[BO RESULT] Best CV R² = {best_bo_r2:.4f}")
    
    best_K, best_n = bo_result.x
    
    bo_weights = {
        "diameter": 0,
        "circularity": 0,
        "compactness": 0, 
        "K": best_K, 
        "n": best_n, 
    }
    all_results.append({**bo_weights, "cv_r2": -bo_result.fun, "source": "BO"})
    print("Best BO weights: diameter: 0", "; circularity: 0" ,"; compactness: 0", "; K: ", best_K, "; n: ", best_n)
    
    def clean_upper_bound(x, step=1.0):
        return step * np.ceil(x / step)

    W_MAX_GLOBAL = clean_upper_bound(
        max(
            bo_weights["circularity"],
            bo_weights["compactness"],
            bo_weights["K"],
            bo_weights["n"],
        ),
        step=1.0
    )

    print("[INFO] Using global weight upper bound:", W_MAX_GLOBAL)




    # ============================================================
    # STAGE 2 — Adaptive grid search
    # ============================================================
    # Purpose:
    #   Build a structured performance landscape around the BO optimum
    #   and historically strong regions, enabling interpretability
    #   and plateau detection.

    print("Starting building the landscape...")
    
    def publication_grid(upper, n=7):
        return np.linspace(0.0, upper, n)


    base_grid = {
        "diameter": [0.0], 
        "circularity": [0.0], 
        "compactness": [0.0], 
        "K": publication_grid(W_MAX_GLOBAL),
        "n": publication_grid(W_MAX_GLOBAL), 
    }


    candidate_points = []

    # 1) BO optimum
    candidate_points.append(bo_weights)



    # 2) Top previous results
    if len(df_prev) > 0:
        top_prev = df_prev.sort_values("cv_r2", ascending=False).head(10)
        for _, r in top_prev.iterrows():
            candidate_points.append({
                "diameter": FIXED_DIAMETER,
                "circularity": r["circularity"],
                "compactness": r["compactness"],
                "K": r["K"],
                "n": r["n"],
            })
    
    def grid_to_weight_dicts(grid):
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]
    
    grid_points = grid_to_weight_dicts(base_grid)
    all_candidates = grid_points + candidate_points
    
    def deduplicate(weights_list, tol=1e-6):
        unique = []
        for w in weights_list:
            if not any(
                all(abs(w[k] - u[k]) < tol for k in w)
                for u in unique
            ):
                unique.append(w)
        return unique

    all_candidates = deduplicate(all_candidates)   
    

    best_score = -np.inf
    best_weights = None
    rows = []

    for weights in tqdm(all_candidates, desc="Unified publication grid"):
        if already_done(weights, df_prev):
            continue

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process)
        row = {**weights, **{f"{k}_descriptor": weights[k] for k in weights},"cv_r2": score}
        rows.append(row)
        all_results.append(row)

        if score > best_score:
            best_score = score
            best_weights = weights.copy()

    df_stage1 = pd.DataFrame(rows)



    
    print("[STAGE 1 BEST]")
    print("R²:", best_score)
    print(best_weights)



    # ============================================================
    # FINAL MODEL TRAINING AND EVALUATION
    # ============================================================
    # The final GP is trained once using the selected weights
    # and evaluated using both a held-out test set and cross-validation.


    final_weights = best_weights if best_weights is not None else bo_weights



    X_final_morph = process.apply_manual_weights(X_vf, {
        "diameter": final_weights["diameter"],
        "circularity": final_weights["circularity"],
        "compactness": final_weights["compactness"],
    })

    X_final_rheo = process.apply_manual_weights_rheo(X_sel_rheo, {
        "K": final_weights["K"],
        "n": final_weights["n"],
    })




    X_final = np.hstack([X_final_morph, X_final_rheo])

    print("X_final shape:", X_final.shape)



    # Split train/test
    indices = np.arange(len(X_final))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_final, y, indices, test_size=0.2, random_state=RANDOM_STATE
    )

    # Optional target scaling
    if SCALE_TARGET:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_train_scaled = y_train

    # Train GP    
    gp = (
        build_gp(RANDOM_STATE)
    )
    print("[INFO] Training Gaussian Process...")
    gp.fit(X_train, y_train_scaled)
    print("[OK] Training done.")

    # Predict
    y_pred_scaled, y_std_scaled = gp.predict(X_test, return_std=True)

    if SCALE_TARGET:
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_std = y_std_scaled * y_scaler.scale_[0]
    else:
        y_pred = y_pred_scaled
        y_std = y_std_scaled

    # Metrics
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    cv_scores = cross_val_score(
        gp, X_final, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
    )

    print("\n[RESULTS]")
    print("RMSE =", rmse)
    print("R² =", r2)
    print("CV mean =", np.mean(cv_scores), "±", np.std(cv_scores))

    # Save predictions
    df_save = pd.DataFrame({
        "sample_name": [sample_names[i] for i in idx_test],
        "true_concentration": y_test,
        "predicted_concentration": y_pred,
        "prediction_std": y_std,
    })
    save_path = CURRENT_DIR / "predictions_diameter_fixed.csv"
    df_save.to_csv(save_path, index=False)
    df_stage1.to_csv("grid_stage1_coarse.csv", index=False)

    print(f"[OK] Saved prediction results to: {save_path}")
    pd.DataFrame(all_results).to_csv(
        LANDSCAPE_FILE,
        index=False
    )














if __name__ == "__main__":
    main()
