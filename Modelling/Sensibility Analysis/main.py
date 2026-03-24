"""
End-to-end pipeline for learning concentration from morphology and rheology
using Gaussian Process Regression.

### Purpose:
This script trains a **Gaussian Process (GP)** to predict concentration from morphological and rheological features.
It includes data loading, preprocessing, hyperparameter optimization, and **Fisher Information Matrix (FIM)** calculation.
The FIM is used to estimate the uncertainty and sensitivity of the model parameters.

### Pipeline Stages:
1. **Load Data**: Morphology, rheology, and concentration data are loaded and aligned.
2. **Data Cleaning**: Invalid or missing entries are filtered out.
3. **Data Augmentation**: Synthetic data is generated using measurement uncertainty.
4. **Feature Preprocessing**: Features are scaled and weighted.
5. **GP Training**: A GP is trained on unweighted features.
6. **Bayesian Optimization (BO)**: Finds optimal feature weights.
7. **FIM Calculation**: Computes the Fisher Information Matrix for parameter uncertainty.
8. **Export Results**: Saves FIM results and predictions.

### Key Files:
- **Input**: Morphology data (in `Data/`), rheology/concentration (Excel file).
- **Output**:
  - `FIM_rotate.csv`: FIM results for parameter uncertainty.
  - `Evaluate/Plots/`: Plots for visualization (if enabled).

### FIM Explanation:
The **Fisher Information Matrix (FIM)** quantifies the amount of information that the data provides about the parameters.
For a GP, the FIM is approximated using the gradient of the mean prediction with respect to the parameters.
The **Cramér-Rao (CR) bound** is derived from the inverse of the FIM and provides a lower bound on the variance of unbiased estimators.

### FIM Formula:
For a GP with mean prediction μ(θ) and variance σ²(θ), the FIM is given by:
\[ F_{ij} = \sum_{k=1}^N \frac{1}{\sigma_k^2} \frac{\partial \mu_k}{\partial \theta_i} \frac{\partial \mu_k}{\partial \theta_j} \]
The CR bound for parameter θ_i is:
\[ \text{CR}_i = \sqrt{(F^{-1})_{ii}} \]
"""

from pathlib import Path
import numpy as np
import pandas as pd

from MatrixGatherer import MorphologyDataLoader
from DataGeneration import DataGeneration
from DataProcessing import Processor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args



# ----------------------------
# Configuration
# ----------------------------
# Paths and input files
CURRENT_DIR = Path(__file__).parent
BASE_PATH = CURRENT_DIR / "Data"
EXCEL_FILE = Path(r"G:\Master\For Leonie\Code u Datenverarbeitung\Modelling\Master Sheet - Change.xlsx")
LANDSCAPE_FILE = CURRENT_DIR / "FIM_rotate.csv"

# Experimental switches
AUGMENT = True          # Enable synthetic data generation from mean ± std
SFBW = True             # Selective feature-by-weight preprocessing
SCALE_TARGET = False    # Whether to standardize target values

# Reproducibility
RANDOM_STATE = 42

# ----------------------------
# FIM configuration
# ----------------------------

ALL_PARAMS = ["diameter", "circularity", "compactness", "K", "n"]

EXCLUDE_PARAMS = []   # <- user control
FIXED_VALUE = 1.0                       # gauge fixing value

ACTIVE_PARAMS = [p for p in ALL_PARAMS if p not in EXCLUDE_PARAMS]


# ----------------------------
# Modeling Helpers
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

# ----------------------------
# FIM Calculation Helpers
# ----------------------------
def predict_with_weights(theta):
    """
    Predicts GP mean and variance for given feature weights.
    - **theta**: Dictionary of feature weights.
    - Returns: Mean (μ) and variance (σ²) of the GP prediction.
    """
    # --- Apply weights to morphology features ---
    morph_weights = {
        "diameter": theta.get("diameter", 0.0),
        "circularity": theta.get("circularity", 0.0),
        "compactness": theta.get("compactness", 0.0),
    }
    X_m = process.apply_manual_weights(X_vf, morph_weights)

    # --- Apply weights to rheology features ---
    rheo_weights = {
        "K": theta.get("K", 0.0),
        "n": theta.get("n", 0.0),
    }
    X_r = process.apply_manual_weights_rheo(X_sel_rheo, rheo_weights)

    # --- Combine features ---
    Xw = np.hstack([X_m, X_r])

    # --- Predict with GP ---
    mu, std = gp.predict(Xw, return_std=True)
    return mu, std**2  # Return mean and variance

def compute_fim_rotating(
    theta_opt,
    active_params,
    eps_base=1e-3,
):
    """
    Computes something similar to the Fisher Information Matrix (FIM) for the GP model and a form of sensitivity analysis.
    The FIM is calculated by fixing one parameter at a time and computing the gradient of the mean prediction.
    The Cramér-Rao (CR) bound is derived from the inverse of the FIM.

    ### FIM Formula:
    \[ F_{ij} = \sum_{k=1}^N \frac{1}{\sigma_k^2} \frac{\partial \mu_k}{\partial \theta_i} \frac{\partial \mu_k}{\partial \theta_j} \]
    where:
    - \(\mu_k\) is the mean prediction for the k-th data point.
    - \(\sigma_k^2\) is the variance for the k-th data point.
    - \(\theta_i\) is the i-th parameter.

    ### CR Bound Formula:
    \[ \text{CR}_i = \sqrt{(F^{-1})_{ii}} \]
    where \(F^{-1}\) is the inverse of the FIM.
    """
    fim_results = []

    for fixed_param in active_params:
        # --- Parameters to optimize (excluding the fixed parameter) ---
        free_params = [p for p in active_params if p != fixed_param]

        # --- Fix one parameter (gauge fixing) ---
        theta_fixed = theta_opt.copy()
        theta_fixed[fixed_param] = FIXED_VALUE  # Fix the parameter to a constant value

        # --- Predict with fixed parameter ---
        mu0, var0 = predict_with_weights(theta_fixed)
        var0 = np.maximum(var0, 1e-6)  # Avoid division by zero

        # --- Compute gradients for free parameters ---
        grads = {}
        for p in free_params:
            # --- Finite difference step size ---
            eps = max(eps_base, 0.01 * abs(theta_fixed[p]))

            # --- Perturb parameter ---
            theta_p = theta_fixed.copy()
            theta_m = theta_fixed.copy()
            theta_p[p] += eps
            theta_m[p] -= eps

            # --- Predict with perturbed parameters ---
            mu_p, _ = predict_with_weights(theta_p)
            mu_m, _ = predict_with_weights(theta_m)

            # --- Central difference gradient ---
            grads[p] = (mu_p - mu_m) / (2 * eps)

        # --- Stack gradients into a matrix (N x M), where N is the number of data points and M is the number of free parameters ---
        G = np.stack([grads[p] for p in free_params], axis=1)

        # --- Initialize FIM ---
        FIM = np.zeros((len(free_params), len(free_params)))

        # --- Compute FIM using the formula: F = G^T * Sigma^-1 * G ---
        for i in range(len(mu0)):
            gi = G[i][:, None]  # Gradient for the i-th data point (M x 1)
            FIM += (gi @ gi.T) / var0[i]  # Add to FIM: (M x 1) * (1 x M) / σ²

        # --- Compute covariance matrix (inverse of FIM) ---
        Cov = np.linalg.pinv(FIM)  # Pseudo-inverse for numerical stability

        # --- Compute CR bounds (square root of diagonal elements of Cov) ---
        CR = np.sqrt(np.diag(Cov))

        # --- Store results ---
        df = pd.DataFrame({
            "parameter": free_params,
            "CR_bound": CR,
            "CR_descriptor_scaled": [CR[i] * 1 for i, p in enumerate(free_params)],
            "fixed_parameter": fixed_param,
            "fixed_value": FIXED_VALUE,
        })

        fim_results.append(df)

    # --- Concatenate results for all fixed parameters ---
    return pd.concat(fim_results, ignore_index=True)





# ----------------------------
# Main
# ----------------------------
def main():
    global process, gp, X_vf, X_sel_rheo

    # ============================================================
    #    1. LOAD RAW DATA
    # ============================================================
    loader = MorphologyDataLoader(BASE_PATH)
    X_mean, X_std, morph_names = loader.load_with_std()
    target_rheo = loader.load_excel_rheology(EXCEL_FILE)
    target_conc = loader.load_excel_concentrations(EXCEL_FILE)

    print(f"[INFO] Morphology samples: {len(morph_names)}")
    print(f"[INFO] Rheology samples  : {len(target_rheo)}")
    print(f"[INFO] Concentration     : {len(target_conc)}")

    # Index map for morphology
    morph_index = {name: i for i, name in enumerate(morph_names)}

    # ==============================================================  
    #    2. CLEAN RHEOLOGY (without dropping!)
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
    final_samples = [
        name for name in morph_names
        if (name in target_conc) and (name in clean_rheology) and (name not in invalid_rheology)
    ]
    all_results = []


    print(f"[INFO] Final aligned sample count: {len(final_samples)}")

    # ==============================================================  
    #    4. EXTRACT ALIGNED DATA
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
    X_sel = process.fit_transform(X)
    X_vf = process.weight_by_vf_vectorized(X_sel) 
    X_sel_rheo = process.fit_transform_rheo(X_rheo)

    # ============================================================
    # 7. TRAIN BASE GP ON UNWEIGHTED FEATURES
    # ============================================================
    print("[INFO] Training base GP on unweighted features...")

    X_base = np.hstack([X_vf, X_sel_rheo])
    gp = build_gp(RANDOM_STATE)
    gp.fit(X_base, y)

    print("[OK] Base GP trained.")


    # ============================================================
    # 8. BAYESIAN OPTIMIZATION FOR OPTIMAL WEIGHTS
    # ============================================================
    space = []
    if "circularity" in ACTIVE_PARAMS:
        space.append(Real(0.0, 5.0, name="circularity"))
    if "compactness" in ACTIVE_PARAMS:
        space.append(Real(0.0, 5.0, name="compactness"))
    if "K" in ACTIVE_PARAMS:
        space.append(Real(0.0, 5.0, name="K"))
    if "n" in ACTIVE_PARAMS:
        space.append(Real(0.0, 5.0, name="n"))


    @use_named_args(space)
    def bo_objective(**params):
        weights = {"diameter": FIXED_VALUE}
        weights.update(params)

        # Fill excluded params with zero
        for p in ALL_PARAMS:
            if p not in weights:
                weights[p] = 0.0

        X_m = process.apply_manual_weights(X_vf, {
            "diameter": weights["diameter"],
            "circularity": weights["circularity"],
            "compactness": weights["compactness"],
        })
        X_r = process.apply_manual_weights_rheo(X_sel_rheo, {
            "K": weights["K"],
            "n": weights["n"],
        })

        Xw = np.hstack([X_m, X_r])
        mu, _ = gp.predict(Xw, return_std=True)

        return -np.var(mu)


    print("[INFO] Running Bayesian Optimization...")
     # --- Run BO ---
    bo_res = gp_minimize(
        bo_objective,
        dimensions=space,
        n_calls=30,
        n_initial_points=10,
        random_state=RANDOM_STATE,
    )

    # --- Extract optimal weights ---
    theta_opt = {"diameter": FIXED_VALUE}
    for name, value in zip([d.name for d in space], bo_res.x):
        theta_opt[name] = value

    # Fill excluded params with zero
    for p in ALL_PARAMS:
        if p not in theta_opt:
            theta_opt[p] = 0.0

    # Finally restrict to ACTIVE_PARAMS
    theta_opt = {k: v for k, v in theta_opt.items() if k in ACTIVE_PARAMS}


    # ============================================================
    # 9. COMPUTE FIM FOR OPTIMAL WEIGHTS
    # ============================================================
    df_fim = compute_fim_rotating(
        theta_opt=theta_opt,
        active_params=ACTIVE_PARAMS,
    )

    # --- Save FIM results ---
    df_fim.to_csv(LANDSCAPE_FILE, index=False)
    print("[OK] Rotating FIM saved to:", LANDSCAPE_FILE)






if __name__ == "__main__":
    main()
