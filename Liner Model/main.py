"""
End-to-end pipeline for learning concentration from morphology and rheology using Ridge Regression.

### Purpose:
This script trains a **linear Ridge regression model** to predict concentration from morphological and rheological features.
It includes data loading, preprocessing, hyperparameter optimization, and evaluation.

### Pipeline Stages:
1. **Load Data**: Morphology, rheology, and concentration data are loaded and aligned.
2. **Data Cleaning**: Invalid or missing entries are filtered out.
3. **Data Augmentation**: Synthetic data is generated using measurement uncertainty.
4. **Feature Preprocessing**: Features are scaled and weighted.
5. **Hyperparameter Search**: Bayesian Optimization (BO) and grid search are used to find optimal feature weights.
6. **Model Training**: The best Ridge model is trained and evaluated.
7. **Export Results**: Predictions, landscapes, and evaluation metrics are saved.

### Key Files:
- **Input**: Morphology data (in `Data/`), rheology/concentration (Excel file).
- **Output**:
  - `predictions_diameter_fixed.csv`: Test set predictions.
  - `parameter_ablation.csv`: Ablation study results.
  - `combined_weight_landscape_diameter_fixed.csv`: Grid search results.
  - `grid_stage1_coarse.csv`: Coarse grid search results.
  - `Evaluate/Plots/`: Plots for visualization (if enabled).

### Configuration:
- **AUGMENT**: Enable/disable synthetic data generation.
- **SFBW**: Enable/disable feature-by-weight preprocessing.
- **SCALE_TARGET**: Standardize target values (concentration).
- **FIXED_DIAMETER**: Fix diameter weight to remove scale invariance.
- **MODEL_TYPE**: Only `"linear"` is supported (Ridge regression).
- **RANDOM_STATE**: For reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time
from functools import lru_cache
from tqdm import tqdm

# --- Custom Modules ---
from MatrixGatherer import MorphologyDataLoader  # Loads morphology/rheology/concentration data
from DataGeneration import DataGeneration        # Generates synthetic data from uncertainty
from DataProcessing import Processor            # Preprocesses features (scaling, weighting)

# --- Scikit-learn ---
from sklearn.linear_model import Ridge           # Linear Ridge regression
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
BASE_PATH = CURRENT_DIR / "Data"                  # Folder for morphology data
EXCEL_FILE = Path(r"G:\Master\For Leonie\Code u Datenverarbeitung\Modelling\Master Sheet - Change.xlsx")  # Rheology/concentration Excel
LANDSCAPE_FILE = CURRENT_DIR / "linear_combined_weight_landscape_diameter_fixed.csv"  # Grid search results

# --- Features ---
ACTIVE_DESCRIPTORS = ["diameter", "circularity", "compactness", "K", "n"]  # Features to use
INACTIVE_DESCRIPTORS = []  # Unused features (if any)

# --- Experimental Switches ---
AUGMENT = True          # Generate synthetic data from mean ± std
SFBW = True             # Feature-by-weight preprocessing
SCALE_TARGET = False    # Standardize target (concentration) values

# --- Reproducibility ---
RANDOM_STATE = 42       # Random seed for reproducibility

# --- Gauge Fixing ---
FIXED_DIAMETER = 1.0    # Fix diameter weight to remove scale invariance

# --- Model Type ---
MODEL_TYPE = "linear"   # Only linear Ridge regression is used




# ----------------------------
# Modeling Helpers
# ----------------------------
def build_linear(random_state=42, alpha=1.0):
    """
    Constructs a linear Ridge regressor.
    - **alpha**: L2 regularization strength (higher = more regularization).
    - **random_state**: Random seed for reproducibility.
    """
    return Ridge(alpha=alpha, random_state=random_state)

def evaluate_weights(weights, X_vf, X_sel_rheo, y, process, model_type="linear"):
    """
    Evaluates feature weights using cross-validated R².
    - Applies weights to morphology and rheology features.
    - Trains a Ridge model and evaluates using 3-fold CV.
    - Returns mean cross-validated R² score.
    """
    W_morph = {k: weights[k] for k in ['diameter', 'circularity', 'compactness']}
    W_rheo = {k: weights[k] for k in ['K', 'n']}

    X_m = process.apply_manual_weights(X_vf, W_morph)
    X_r = process.apply_manual_weights_rheo(X_sel_rheo, W_rheo)
    Xc = np.hstack([X_m, X_r])

    model = build_linear(RANDOM_STATE)
    scores = cross_val_score(
        model, Xc, y,
        cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2"
    )
    return scores.mean()

# ----------------------------
# Grid Search
# ----------------------------
def already_done(weights, df_prev, tol=1e-6):
    """Checks if a weight combination was already evaluated."""
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
    - **Output**: DataFrame of results, saved to `LANDSCAPE_FILE`.
    """
    start_time = time.time()
    results = []
    best_score = -np.inf
    best_weights = None

    keys, values = zip(*grid.items())
    combinations = list(product(*values))
    total = len(combinations)

    print(f"[GRID] Evaluating up to {total} combinations (skipping duplicates)")

    for i, combination in enumerate(tqdm(combinations, desc="Grid search"), 1):
        weights = dict(zip(keys, combination))
        if already_done(weights, df_prev):
            continue

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process, MODEL_TYPE)
        row = {**weights, **{f"{k}_descriptor": weights[k] for k in weights}, "cv_r2": score}
        results.append(row)
        all_results.append(row)

        if score > best_score:
            best_score = score
            best_weights = weights.copy()

        if verbose and (i % 10 == 0 or i == 1):
            elapsed = time.time() - start_time
            print(f"[GRID {i}/{total}] elapsed={elapsed:.1f}s, cv_r2={score:.4f}")

    return best_score, best_weights, pd.DataFrame(results)







# ----------------------------
# Main Code
# ----------------------------
def main():
    # ============================================================
    #    1) LOAD RAW DATA
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
    #    2) CLEAN RHEOLOGY (without dropping!)
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
    #    3) Compute FINAL valid sample list (only AFTER cleaning)
    # ==============================================================
    final_samples = [
        name for name in morph_names
        if (name in target_conc) and (name in clean_rheology) and (name not in invalid_rheology)
    ]
    all_results = []

    print(f"[INFO] Final aligned sample count: {len(final_samples)}")

    # ==============================================================
    #    4) Extract aligned morphology, rheology, concentration
    # ==============================================================

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

    # ============================================================
    # 6. FEATURE PREPROCESSING
    # ============================================================
    process = Processor()
    X_sel = process.fit_transform(X)  # Scale morphology features
    X_vf = process.weight_by_vf_vectorized(X_sel)  # Weight by volume fraction
    X_sel_rheo = process.fit_transform_rheo(X_rheo)  # Scale rheology features

    # ============================================================
    # 7. LOAD PREVIOUS RESULTS (if any)
    # ============================================================
    RESULTS_FILE = LANDSCAPE_FILE

    if RESULTS_FILE.exists():
        df_prev = pd.read_csv(RESULTS_FILE)
        print(f"[INFO] Loaded {len(df_prev)} previous evaluations")
    else:
        df_prev = pd.DataFrame()

    # ============================================================
    # 8. PARAMETER ABLATION STUDY
    # ============================================================
    print("\n[ABLATION] Starting parameter ablation study...")

    descriptors = ACTIVE_DESCRIPTORS
    ablation_rows = []

    # ------------------------------------------------------------
    # 8. A — Single-descriptor models (pure 1D)
    # ------------------------------------------------------------
    print("[ABLATION A] Single-descriptor models")

    for d in descriptors:
        weights = {k: 0.0 for k in ACTIVE_DESCRIPTORS + INACTIVE_DESCRIPTORS}
        weights[d] = 1.0  # arbitrary nonzero scale

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process, model_type=MODEL_TYPE)

        ablation_rows.append({
            "mode": "single_descriptor",
            "ablated": None,
            "active": d,
            "cv_r2": score
        })

        print(f"  [1D] {d:12s} → R² = {score:.4f}")

    # ------------------------------------------------------------
    # 8.B — Leave-one-out BO (multi-D with gauge fixing)
    # ------------------------------------------------------------
    print("\n[ABLATION B] Leave-one-out BO")

    for ablated in ACTIVE_DESCRIPTORS:
        if ablated == "diameter":
            fixed_diameter = 0.0
        else:
            fixed_diameter = FIXED_DIAMETER

        space = []
        param_names = []

        for p in ACTIVE_DESCRIPTORS:
            if p == ablated or p == "diameter":
                continue
            space.append(Real(0.0, 5.0, name=p))

        @use_named_args(space)
        def loo_objective(**params):
            weights = {k: 0.0 for k in ACTIVE_DESCRIPTORS + INACTIVE_DESCRIPTORS}
            weights.update(params)
            weights["diameter"] = fixed_diameter
            weights[ablated] = 0.0

            score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process, model_type=MODEL_TYPE)
            return -score

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
    df_ablation.to_csv(CURRENT_DIR / "linear_parameter_ablation.csv", index=False)

    print("[ABLATION] Results saved to parameter_ablation.csv")

    # ============================================================
    # 9. BAYESIAN OPTIMIZATION
    # ============================================================
    print("Starting the Bayesian Optimizer...")
    space = [Real(0, 5.0, name=d) for d in ["circularity", "compactness", "K", "n"]]

    @use_named_args(space)
    def objective(circularity, compactness, K, n):
        weights = {
            "diameter": FIXED_DIAMETER,
            "circularity": circularity,
            "compactness": compactness,
            "K": K,
            "n": n
        }

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process, model_type=MODEL_TYPE)

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

    best_c, best_k, best_K, best_n = bo_result.x

    bo_weights = {
        "diameter": FIXED_DIAMETER,
        "circularity": best_c,
        "compactness": best_k,
        "K": best_K,
        "n": best_n,
    }
    all_results.append({**bo_weights, "cv_r2": -bo_result.fun, "source": "BO"})
    print("Best BO weights: diameter: ", FIXED_DIAMETER, "; circularity: ", best_c, "; compactness: ", best_k, "; K: ", best_K, "; n: ", best_n)

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
    # 10. ADAPTIVE GRID SEARCH
    # ============================================================
    print("Starting building the landscape...")

    def publication_grid(upper, n=7):
        return np.linspace(0.0, upper, n)

    base_grid = {
        "diameter": [FIXED_DIAMETER],
        "circularity": publication_grid(W_MAX_GLOBAL),
        "compactness": publication_grid(W_MAX_GLOBAL),
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

        score = evaluate_weights(weights, X_vf, X_sel_rheo, y, process, model_type=MODEL_TYPE)
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
    # 11. FINAL MODEL TRAINING
    # ============================================================
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

    # Train Ridge
    model = build_linear(RANDOM_STATE)
    print("[INFO] Training Ridge Regression...")
    model.fit(X_train, y_train_scaled)
    print("[OK] Training done.")

    y_pred_scaled = model.predict(X_test)

    if SCALE_TARGET:
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred = y_pred_scaled

    # Metrics
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    cv_scores = cross_val_score(
        model, X_final, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
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
    })
    save_path = CURRENT_DIR / "linear_predictions_diameter_fixed.csv"
    df_save.to_csv(save_path, index=False)

    print(f"[OK] Saved prediction results to: {save_path}")
    pd.DataFrame(all_results).to_csv(
        LANDSCAPE_FILE,
        index=False
    )

if __name__ == "__main__":
    main()
