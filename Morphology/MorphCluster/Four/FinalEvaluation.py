import os
from pathlib import Path
import pandas as pd
from .Plot import PSDPlotter

class MatrixEvaluator:
    """
    Walk through all subfolders containing 'PSD_clustered_summary.csv',
    and compute per-cluster statistics for the chosen cluster type and count:
      - mean diameter
      - std of diameter
      - total volume fraction
      - mean circularity (Irregularity)
      - number of particles
    """

    def __init__(self, base_path: Path, cluster_type: str, cluster_count: int):
        self.base_path = Path(base_path)
        self.cluster_type = cluster_type.strip().capitalize()  # "Volume" or "Number"
        self.cluster_count = int(cluster_count)

    def clean_folder(self, folder: Path):
        """Delete all non-essential files to prevent stacking."""
        for file in folder.iterdir():
            if file.is_file() and file.name not in {"combined_data.csv", "PSD_clustered_summary.csv"}:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"[WARN] Could not delete {file}: {e}")

    def process(self):
        print(f"[INFO] Starting final PSD matrix evaluation in: {self.base_path}")
        print(f"[INFO] Using clustering: {self.cluster_type}-based, k={self.cluster_count}")

        for root, _, files in os.walk(self.base_path):
            root_path = Path(root)

            if "PSD_clustered_summary.csv" not in files:
                continue

            file_path = root_path / "PSD_clustered_summary.csv"
            print(f"[DBG] Found data file: {file_path.name} in {root_path}")

            # --- Clean up old results ---
            self.clean_folder(root_path)

           # --- Load data ---
            df = pd.read_csv(file_path, skiprows=1)
            df.columns = df.columns.str.strip().str.lower()

            # --- Select correct cluster column ---
            cluster_col = f"cluster_{self.cluster_type.lower()}_k{self.cluster_count}"

            if cluster_col not in df.columns:
                print(f"[ERR] Missing column: {cluster_col}. Skipping {file_path.name}")
                continue

            if df[cluster_col].isna().all():
                print(f"[WARN] No cluster data in {file_path.name}. Skipping.")
                continue

            df = df.copy()
            df["cluster"] = df[cluster_col].astype(int)

            # --- Check required columns ---
            required = ["diameter [mym]", "volume fraction", "irregularity [-]", "diameter ratio [-]"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"[ERR] Missing required columns {missing} in {file_path.name}. Skipping this file.")
                continue

            # --- Group by cluster ---
            grouped = df.groupby("cluster").agg(
                Mean_Diameter=("diameter [mym]", "mean"),
                Std_Diameter=("diameter [mym]", "std"),
                Total_VolumeFraction=("volume fraction", "sum"),
                Mean_Circularity=("irregularity [-]", "mean"),
                Std_Circularity=("irregularity [-]", "std"),
                Mean_Compactness=("diameter ratio [-]", "mean"),
                Std_Compactness=("diameter ratio [-]", "std"),
                Particle_Count=("diameter [mym]", "count")
            ).reset_index()


            # --- Save output ---
            out_file = root_path / f"PSD_cluster_summary_final_k{self.cluster_count}_{self.cluster_type}.csv"
            grouped.to_csv(out_file, index=False)
            print(f"[OK] Saved: {out_file.name}")

            plotter = PSDPlotter(root_path / "PSD_clustered_summary.csv", self.cluster_type, self.cluster_count)
            plotter.run()
