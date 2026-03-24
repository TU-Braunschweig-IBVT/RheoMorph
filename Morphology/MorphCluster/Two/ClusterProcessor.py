import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class PSDClusterProcessor:
    """
    Walk through all subfolders containing 'combined_data.csv',
    and cluster each PSD in two ways:
      1. Based on Volume Fraction (volume-weighted PSD)
      2. Based on diameter count (number-weighted PSD)

    Everything is saved in ONE combined file per PSD.
    """

    def __init__(self, base_path: Path, bins: int = 25):
        self.base_path = Path(base_path)
        self.bins = bins

    def clean_folder(self, folder: Path):
        """Delete all files except 'combined_data.csv' to prevent stacking."""
        for file in folder.iterdir():
            if file.is_file() and file.name != "combined_data.csv":
                try:
                    file.unlink()
                except Exception as e:
                    print(f"[WARN] Could not delete {file}: {e}")

    def assign_histogram_bins(self, df: pd.DataFrame):
        """
        Assign each pellet to a diameter histogram bin
        using linear binning (same logic as np.histogram).
        """
        diam = df["Diameter [mym]"].to_numpy()

        bin_edges = np.linspace(diam.min(), diam.max(), self.bins + 1)

        # np.digitize returns bins in [1, bins]
        bin_index = np.digitize(diam, bin_edges, right=False) - 1

        # Clamp edge case (max diameter)
        bin_index = np.clip(bin_index, 0, self.bins - 1)

        df["Histogram_Bin"] = bin_index
        df["Histogram_Bin_Lower"] = bin_edges[bin_index]
        df["Histogram_Bin_Upper"] = bin_edges[bin_index + 1]

        return df, bin_edges


    def save_histogram_input(self, df: pd.DataFrame, bin_edges: np.ndarray, folder: Path):
        """
        Save a clean PSD file containing explicit histogram bin assignments.
        """
        hist_df = pd.DataFrame({
            "Diameter [mym]": df["Diameter [mym]"],
            "Histogram_Bin": df["Histogram_Bin"],
            "Bin_Lower [mym]": df["Histogram_Bin_Lower"],
            "Bin_Upper [mym]": df["Histogram_Bin_Upper"],
            "Volume [mym^3]": df["Volume [mym^3]"],
            "Volume Fraction": df["Volume Fraction"],
        })

        out_file = folder / "PSD_histogram_input.csv"
        with open(out_file, "w", newline="") as f:
            f.write("# PSD histogram reconstruction file\n")
            f.write(f"# Linear diameter binning\n")
            f.write(f"# Number of bins: {self.bins}\n")
            f.write("# Histogram_Bin is zero-based\n")
            hist_df.to_csv(f, index=False)

        print(f"[OK] Saved histogram input to {out_file.name}")
    
    def save_histogram_bins(self, df: pd.DataFrame, folder: Path):
        """
        Save aggregated histogram bins with number- and volume-weighted PSD.
        """
        grouped = df.groupby("Histogram_Bin")

        bin_df = grouped.agg(
            Bin_Lower_mym=("Histogram_Bin_Lower", "first"),
            Bin_Upper_mym=("Histogram_Bin_Upper", "first"),
            Pellet_Count=("Histogram_Bin", "size"),
            Volume_Sum_mym3=("Volume [mym^3]", "sum"),
            Volume_Fraction=("Volume Fraction", "sum"),
        ).reset_index()

        # Number-weighted PSD
        total_count = len(df)
        bin_df["Number_Fraction"] = bin_df["Pellet_Count"] / total_count

        # Sanity checks
        bin_df["Volume_Fraction"] = bin_df["Volume_Fraction"].fillna(0)
        bin_df["Number_Fraction"] = bin_df["Number_Fraction"].fillna(0)

        out_file = folder / "PSD_histogram_bins.csv"
        with open(out_file, "w", newline="") as f:
            f.write("# Aggregated PSD histogram bins\n")
            f.write("# Number_Fraction sums to 1\n")
            f.write("# Volume_Fraction sums to 1\n")
            bin_df.to_csv(f, index=False)

        print(f"[OK] Saved histogram bins to {out_file.name}")


    def save_psd_plot(self, folder: Path):
        """
        Save a volume-weighted PSD histogram plot.
        """
        bins_file = folder / "PSD_histogram_bins.csv"
        if not bins_file.exists():
            print(f"[WARN] No histogram bins found in {folder}")
            return

        bin_df = pd.read_csv(bins_file, comment="#")

        # Bin centers and widths
        bin_centers = 0.5 * (bin_df["Bin_Lower_mym"] + bin_df["Bin_Upper_mym"])
        bin_widths = bin_df["Bin_Upper_mym"] - bin_df["Bin_Lower_mym"]

        plt.figure(figsize=(7, 5))
        plt.bar(
            bin_centers,
            bin_df["Volume_Fraction"],
            width=bin_widths,
            align="center",
            edgecolor="black"
        )

        plt.xlabel("Diameter [µm]")
        plt.ylabel("Volume fraction [-]")
        plt.title("Volume-weighted Particle Size Distribution")

        plt.tight_layout()

        out_file = folder / "PSD_volume_weighted.png"
        plt.savefig(out_file, dpi=300)
        plt.close()

        print(f"[OK] Saved PSD plot to {out_file.name}")



    def process(self):
        print(f"[INFO] Starting individual PSD clustering in: {self.base_path}")

        # Walk through the entire folder tree
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)

            # Only process folders that contain combined_data.csv
            if "combined_data.csv" not in files:
                continue

            # Remembers the folder path that is checked out at the moment
            file_path = root_path / "combined_data.csv"
            print(f"[DBG] Found data file: {file_path}")

            # Clean up old results
            self.clean_folder(root_path)

            # --- Load data ---
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # remove trailing spaces (The Volume got a blank space in the end)

            # Check required columns -> Volume and Diameter
            if "Diameter [mym]" not in df.columns or "Volume [mym^3]" not in df.columns:
                print(f"[WARN] Skipping {file_path.name} — missing required columns.")
                continue

            # Sort and compute volume fraction
            df = df.sort_values(by="Diameter [mym]").reset_index(drop=True)
            total_volume = df["Volume [mym^3]"].sum()
            # making sure there is actually a real volume
            if total_volume == 0 or pd.isna(total_volume):
                print(f"[WARN] Skipping {file_path.name} — total volume is zero or invalid.")
                continue
            # Compute volume fraction
            df["Volume Fraction"] = df["Volume [mym^3]"] / total_volume


            # Assign histogram bins
            df, bin_edges = self.assign_histogram_bins(df)
            # Save clean histogram input (NO clustering)
            self.save_histogram_input(df, bin_edges, root_path)
            # Save aggregated histogram bins
            self.save_histogram_bins(df, root_path)
            # Save PSD plot
            self.save_psd_plot(root_path)



            # Store all cluster summaries (so that it can be saved in one file)
            summary_rows = []

            #           ~~ Clustering ~~

            # --- (1) Volume Fraction clustering ---
            for k in range(2, 8):
                try:
                    print(f"[INFO] Clustering Volume Fraction (k={k}) for {file_path.name}...")

                    #y_values = df[["Volume Fraction"]].to_numpy()
                    #kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    #labels = kmeans.fit_predict(y_values)
                    #df[f"Cluster_Volume_k{k}"] = labels
                    df["KMeans_Input_Volume"] = df["Volume Fraction"]
                    for k in range(2, 8):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(df[["KMeans_Input_Volume"]])
                        df[f"Cluster_Volume_k{k}"] = labels



                except Exception as e:
                    print(f"[ERROR] Volume-fraction clustering failed for k={k} in {file_path.name}: {e}")

            # --- (2) Diameter-only clustering ---
            for k in range(2, 8):
                try:
                    print(f"[INFO] Clustering by diameter frequency (k={k}) for {file_path.name}...")

                    # Cluster along index (frequency of occurrence)
                    #y_values = np.arange(len(df)).reshape(-1, 1)
                    #kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    #labels = kmeans.fit_predict(y_values)
                    #df[f"Cluster_Number_k{k}"] = labels
                    df["KMeans_Input_Number"] = np.arange(len(df))
                    for k in range(2, 8):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(df[["KMeans_Input_Number"]])
                        df[f"Cluster_Number_k{k}"] = labels


                except Exception as e:
                    print(f"[ERROR] Diameter-only clustering failed for k={k} in {file_path.name}: {e}")

            # --- (3) Save all results in one file ---
            summary_df = pd.DataFrame(summary_rows)

            output_file = root_path / "PSD_clustered_summary.csv"
            with open(output_file, "w", newline="") as f:
                f.write("# Original PSD data with all cluster assignments\n")
                df.to_csv(f, index=False)

            print(f"[OK] Saved combined results to {output_file.name}\n")

        print("[DONE] Individual PSD clustering complete.")
