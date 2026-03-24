import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

class ClusterValidator:
    """
    Validates PSD clustering using Silhouette and RMSSTD metrics.
    Iterates through all PSD_clustered_summary.csv files
    and aggregates results for both Volume Fraction and Number-based clustering.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        # Folder for evaluation output (same level as Output/)
        self.eval_path = self.base_path.parent / "Evaluation Clustering"
        self.eval_path.mkdir(exist_ok=True)

        # Containers for aggregated results
        self.results = {
            "Volume Fraction": {k: {"silhouette": [], "rmsstd": []} for k in range(2, 8)},
            "Number": {k: {"silhouette": [], "rmsstd": []} for k in range(2, 8)},
        }

    def process(self):
        print(f"[INFO] Starting clustering validation in: {self.base_path}")

        # Walk through the entire folder tree
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            if "PSD_clustered_summary.csv" not in files:
                continue


            # Remembers the folder path that is checked out at the moment
            file_path = root_path / "PSD_clustered_summary.csv"
            print(f"[DBG] Reading data from: {file_path}")

            # Read file (ignore commented header lines)
            df = pd.read_csv(file_path, comment="#")
            df.columns = df.columns.str.strip()

            # Ensure required columns
            if "Diameter [mym]" not in df.columns:
                print(f"[WARN] Skipping {file_path.name}: missing diameter column.")
                continue

            # --- (1) Validate Volume Fraction Clusterings ---
            for k in range(2, 8):
                label_col = f"Cluster_Volume_k{k}"
                if label_col not in df.columns:
                    continue

                y = df["Volume Fraction"].to_numpy().reshape(-1, 1)
                labels = df[label_col].to_numpy()

                if len(np.unique(labels)) < 2:
                    continue

                try:
                    sil = silhouette_score(y, labels)
                    rmsstd = self.compute_rmsstd(y, labels)
                    self.results["Volume Fraction"][k]["silhouette"].append(sil)
                    self.results["Volume Fraction"][k]["rmsstd"].append(rmsstd)
                except Exception as e:
                    print(f"[WARN] Skipping k={k} Volume Fraction in {file_path.name}: {e}")

            # --- (2) Validate Number-based Clusterings ---
            for k in range(2, 8):
                label_col = f"Cluster_Number_k{k}"
                if label_col not in df.columns:
                    continue

                # Use diameter itself as value to check cluster compactness
                y = df["Diameter [mym]"].to_numpy().reshape(-1, 1)
                labels = df[label_col].to_numpy()

                if len(np.unique(labels)) < 2:
                    continue

                try:
                    sil = silhouette_score(y, labels)
                    rmsstd = self.compute_rmsstd(y, labels)
                    self.results["Number"][k]["silhouette"].append(sil)
                    self.results["Number"][k]["rmsstd"].append(rmsstd)
                except Exception as e:
                    print(f"[WARN] Skipping k={k} Number-based in {file_path.name}: {e}")

        self.save_results()
        print("[DONE] Cluster validation complete.")

    @staticmethod
    def compute_rmsstd(values, labels):
        """Compute the Root Mean Square Standard Deviation for cluster compactness."""
        clusters = np.unique(labels)
        total_ss = 0
        n_total = len(values)
        for cluster in clusters:
            cluster_vals = values[labels == cluster]
            if len(cluster_vals) > 1:
                mean = np.mean(cluster_vals)
                total_ss += np.sum((cluster_vals - mean) ** 2)
        return np.sqrt(total_ss / (n_total - len(clusters)))

    def save_results(self):
        """Save mean silhouette and RMSSTD for each clustering setup."""
        rows = []
        for method, k_dict in self.results.items():
            for k, metrics in k_dict.items():
                sil_scores = metrics["silhouette"]
                rms_scores = metrics["rmsstd"]
                rows.append({
                    "Type": method,
                    "k": k,
                    "Mean Silhouette": np.mean(sil_scores) if sil_scores else np.nan,
                    "Mean RMSSTD": np.mean(rms_scores) if rms_scores else np.nan,
                    "Counted Files": len(sil_scores)
                })

        df_out = pd.DataFrame(rows)
        out_path = self.eval_path / "Evaluation_Results.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Saved evaluation results to {out_path}")
