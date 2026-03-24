# file: MatrixGatherer.py
"""
MorphologyDataLoader
- Loads clustered PSD CSVs (per-sample) and returns flattened feature vectors
  with both means and stds available for augmentation.
- Provides a helper to load concentration means + stds from the Excel "master sheet".
"""

from pathlib import Path
import pandas as pd
import numpy as np


class MorphologyDataLoader:
    def __init__(self, base_path: Path):
        """
        Parameters
        ----------
        base_path : Path-like
            Directory containing per-sample subfolders with CSV files.
        cluster_count : int
            Expected number of clusters per sample (makes flattening/reshaping deterministic).
        """
        self.base_path = Path(base_path)
        self.cluster_count = 3
        self.ALLOWED_INNER_FOLDERS = {"7"}      # or: {"5", "7"} Decide here which day you wanna look at
    # ------------------------------------------------------------------
    # SD loader: returns average sd/mean for each category to use as a fallback if sd is missing (see the code below)
    # ------------------------------------------------------------------
    def compute_global_sd_ratios(self):
        """
        Scan all morphology CSVs and compute average (std / mean)
        for each metric across all samples and clusters.
        """
        

        pattern = f"PSD_cluster_summary_final_k{self.cluster_count}_Volume.csv"

        metrics = {
            "Diameter":      ("Mean_Diameter",      "Std_Diameter"),
            "Circularity":   ("Mean_Circularity",   "Std_Circularity"),
            "Compactness":   ("Mean_Compactness",   "Std_Compactness"),
        }

        ratios = {k: [] for k in metrics}

        for outer in self.base_path.iterdir():
            if not outer.is_dir():
                continue

            for inner in outer.iterdir():
                if not inner.is_dir():
                    continue
                if inner.name not in self.ALLOWED_INNER_FOLDERS:
                    continue

                file_path = inner / pattern
                if not file_path.exists():
                    continue

                df = pd.read_csv(file_path, na_values=["None", "none", "", "NaN"])

                for name, (mean_col, std_col) in metrics.items():
                    if mean_col not in df.columns or std_col not in df.columns:
                        continue

                    valid = df[[mean_col, std_col]].dropna()
                    valid = valid[valid[mean_col] != 0]

                    ratios[name].extend(
                        (valid[std_col] / valid[mean_col]).tolist()
                    )

        # Compute averages (fallback if absolutely no data)
        avg_ratios = {}
        for k, vals in ratios.items():
            if len(vals) == 0:
                avg_ratios[k] = 0.3  # conservative fallback
            else:
                avg_ratios[k] = float(np.mean(vals))

        return avg_ratios

    # ------------------------------------------------------------------
    # New loader: returns means + stds (flattened) + sample names
    # ------------------------------------------------------------------
    def load_with_std(self):
        """
        Strict morphology loader:
        - Missing STD → replaced with zero (allowed)
        - Missing any MEAN or VolumeFraction → sample skipped (fatal for that sample)
        - Missing cluster row → sample skipped
        """
        sd_ratios = self.compute_global_sd_ratios()

        pattern = f"PSD_cluster_summary_final_k{self.cluster_count}_Volume.csv"

        means_list = []
        stds_list = []
        sample_names = []

        metric_mean_cols = ["Mean_Diameter", "Mean_Circularity", "Mean_Compactness"]
        metric_std_cols  = ["Std_Diameter",  "Std_Circularity",  "Std_Compactness"]
        vf_col = "Total_VolumeFraction"

        for outer in sorted(self.base_path.iterdir()):
            if not outer.is_dir():
                continue
            for inner in sorted(outer.iterdir()):
                if not inner.is_dir():
                    continue
                if inner.name not in self.ALLOWED_INNER_FOLDERS:
                    continue

                file_path = inner / pattern
                if not file_path.exists():
                    continue

                df = pd.read_csv(file_path, na_values=["None", "none", "", "NaN"])

                # --- Fatal check #1: Required columns must exist ---
                required_columns = metric_mean_cols + metric_std_cols + [vf_col]
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    print(f"[ERROR] Missing columns {missing_cols} in {outer}/{inner} — skipping sample.")
                    continue

                # --- Sort by diameter (or any metric you prefer) ---
                df = df.sort_values(by="Mean_Diameter", ascending=False)

                # --- Fatal check #2: Must have all clusters ---
                if len(df) < self.cluster_count:
                    print(f"[ERROR] {outer}/{inner} has only {len(df)} clusters; expected {self.cluster_count}. Skipping.")
                    continue

                df = df.head(self.cluster_count).reset_index(drop=True)

                # --- Fatal check #3: Any missing *means or volume fractions* → reject ---
                if df[metric_mean_cols + [vf_col]].isna().any().any():
                    print(f"[ERROR] Missing MEAN or VF values in {outer}/{inner} — skipping sample.")
                    continue

                # -----------------------------------------------------------------
                # Missing STD values are allowed → replaced with average sd//mean * current mean
                # -----------------------------------------------------------------
                df_std = df[metric_std_cols].copy()

                for mean_col, std_col, metric_name in zip(
                    metric_mean_cols,
                    metric_std_cols,
                    ["Diameter", "Circularity", "Compactness"]
                ):
                    missing = df_std[std_col].isna()
                    if missing.any():
                        df_std.loc[missing, std_col] = (
                            df.loc[missing, mean_col] * sd_ratios[metric_name]
                        )


                # Convert to arrays
                means = df[metric_mean_cols].to_numpy()      # shape (k,3)
                stds  = df_std.to_numpy()                    # shape (k,3)
                vf    = df[vf_col].to_numpy()                # shape (k,)

                # Build flattened matrices: (k clusters × 4 features)
                mat_mean = np.hstack([means, vf[:, None]])
                mat_std  = np.hstack([stds,  np.zeros((self.cluster_count, 1))])  # VF std = 0 always

                means_list.append(mat_mean.flatten())
                stds_list.append(mat_std.flatten())
                sample_names.append(f"{outer.name}.{inner.name}")

        # Convert to arrays
        X_means = np.array(means_list)
        X_stds  = np.array(stds_list)

        return X_means, X_stds, sample_names

    # -------------------Rheology sd ratio generation-----------------------------------------------
    def compute_rheology_sd_ratios(
        self,
        excel_file,
        sample_prefix="DOE ",
        mean1_col=20,
        sd1_col=22,
        mean2_col=21,
        sd2_col=23,
    ):
        """
        Compute average SD/Mean ratios for two rheological metrics.
        """

        df = pd.read_excel(excel_file)

        ratios_1 = []
        ratios_2 = []

        for _, row in df.iterrows():
            cell = str(row[0]) if not pd.isna(row[0]) else ""
            if not (isinstance(cell, str) and cell.startswith(sample_prefix)):
                continue

            # metric 1
            try:
                mean1 = float(row.iloc[mean1_col])
                sd1 = float(row.iloc[sd1_col])
                if mean1 > 0 and sd1 > 0 and not np.isnan(sd1):
                    ratios_1.append(sd1 / mean1)
            except Exception:
                pass

            # metric 2
            try:
                mean2 = float(row.iloc[mean2_col])
                sd2 = float(row.iloc[sd2_col])
                if mean2 > 0 and sd2 > 0 and not np.isnan(sd2):
                    ratios_2.append(sd2 / mean2)
            except Exception:
                pass

        # Fallbacks if nothing valid exists
        ratio1 = float(np.mean(ratios_1)) if ratios_1 else 0.3
        ratio2 = float(np.mean(ratios_2)) if ratios_2 else 0.3

        return ratio1, ratio2

    # -------------------Rheology loading-----------------------------------------------
    def load_excel_rheology(self, excel_file):
        df = pd.read_excel(excel_file)
        target_map = {}
        sample_prefix = "DOE "

        mean1_col, sd1_col = 20, 22
        mean2_col, sd2_col = 21, 23

        # 🔹 compute data-driven ratios once
        ratio1, ratio2 = self.compute_rheology_sd_ratios(
            excel_file,
            sample_prefix,
            mean1_col,
            sd1_col,
            mean2_col,
            sd2_col,
        )

        for _, row in df.iterrows():
            cell = str(row[0]) if not pd.isna(row[0]) else ""
            if isinstance(cell, str) and cell.startswith(sample_prefix):
                name = cell[len(sample_prefix):]

                # --- Mean 1 (fatal) ---
                try:
                    mean1 = float(row.iloc[mean1_col])
                except Exception:
                    print(f"[WARN] Mean1 missing/invalid for {name}; skipping.")
                    continue

                # --- SD 1 (imputed if needed) ---
                try:
                    sd1_raw = row.iloc[sd1_col]
                    sd1 = float(sd1_raw) if not pd.isna(sd1_raw) else np.nan
                    if np.isnan(sd1) or sd1 == 0:
                        sd1 = ratio1 * mean1
                except Exception:
                    sd1 = ratio1 * mean1

                # --- Mean 2 (optional) ---
                try:
                    mean2 = float(row.iloc[mean2_col])
                except Exception:
                    mean2 = np.nan

                # --- SD 2 (imputed if possible) ---
                try:
                    sd2_raw = row.iloc[sd2_col]
                    sd2 = float(sd2_raw) if not pd.isna(sd2_raw) else np.nan
                    if np.isnan(sd2) or sd2 == 0:
                        sd2 = ratio2 * mean2 if not np.isnan(mean2) else np.nan
                except Exception:
                    sd2 = ratio2 * mean2 if not np.isnan(mean2) else np.nan

                target_map[name] = [mean1, mean2, sd1, sd2]

        return target_map

   
    # -------------------Fallback SD for Concentration-----------------------------------------------
    def compute_concentration_sd_ratio(
        self,
        excel_file,
        sample_prefix="DOE ",
        mean_col_index=3,
        sd_col_index=4,
    ):
        """
        Compute average SD/Mean ratio for pigment concentration
        from all valid rows in the Excel sheet.
        """

        df = pd.read_excel(excel_file)
        ratios = []

        for _, row in df.iterrows():
            cell = str(row[0]) if not pd.isna(row[0]) else ""
            if not (isinstance(cell, str) and cell.startswith(sample_prefix)):
                continue

            try:
                mean = float(row.iloc[mean_col_index])
                sd = float(row.iloc[sd_col_index])
            except Exception:
                continue

            if mean > 0 and sd > 0 and not np.isnan(sd):
                ratios.append(sd / mean)

        if len(ratios) == 0:
            # conservative fallback if no valid SDs exist
            return 0.3

        return float(np.mean(ratios))

    # -------------------Concentration loading-----------------------------------------------
    def load_excel_concentrations(self, excel_file):
        df = pd.read_excel(excel_file)
        target_map = {}

        sample_prefix = "DOE "
        mean_col_index = 3
        sd_col_index = 4

        # 🔹 compute data-driven fallback once
        sd_ratio = self.compute_concentration_sd_ratio(
            excel_file,
            sample_prefix,
            mean_col_index,
            sd_col_index,
        )

        for _, row in df.iterrows():
            cell = str(row[0]) if not pd.isna(row[0]) else ""
            if isinstance(cell, str) and cell.startswith(sample_prefix):
                name = cell[len(sample_prefix):]

                # mean (fatal)
                try:
                    mean = float(row.iloc[mean_col_index])
                except Exception:
                    print(f"[WARN] Mean missing/invalid for {name}; skipping.")
                    continue
                # sd (imputed if missing)
                try:
                    sd_raw = row.iloc[sd_col_index]
                    sd = float(sd_raw) if not pd.isna(sd_raw) else np.nan
                    if np.isnan(sd) or sd == 0:
                        sd = sd_ratio * mean
                except Exception:
                    sd = sd_ratio * mean

                target_map[name] = (mean, float(sd))

        return target_map