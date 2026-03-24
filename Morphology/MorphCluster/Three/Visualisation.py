import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ClusterVisualizer:
    """
    Reads the clustering evaluation results and visualizes:
      - Silhouette Score vs. Cluster Count (Number & Volume)
      - RMSSTD vs. Cluster Count (Number & Volume)
    """

    def __init__(self, eval_folder: Path):
        self.eval_folder = Path(eval_folder)
        self.output_dir = self.eval_folder
        # Adapted to your filename
        self.eval_file = self.eval_folder / "Evaluation_Results.csv"

    def visualize(self):
        print(f"[INFO] Starting visualization using: {self.eval_file}")

        if not self.eval_file.exists():
            print(f"[ERR] Evaluation file not found at: {self.eval_file}")
            return

        # --- Load evaluation data ---
        df = pd.read_csv(self.eval_file)
        df.columns = df.columns.str.strip()  # remove trailing spaces

        # --- Rename columns for consistency ---
        column_map = {
            "k": "Clusters",
            "Mean Silhouette": "Silhouette",
            "Mean RMSSTD": "RMSSTD",
        }
        df.rename(columns=column_map, inplace=True)

        required_cols = {"Type", "Clusters", "Silhouette", "RMSSTD"}
        if not required_cols.issubset(df.columns):
            print(f"[ERR] Missing required columns. Found: {df.columns.tolist()}")
            return

        # Ensure numeric conversion
        for col in ["Clusters", "Silhouette", "RMSSTD"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Split by Type
        num_df = df[df["Type"].str.strip() == "Number"]
        vol_df = df[df["Type"].str.strip() == "Volume Fraction"]

        # --- Plot 1 & 2: Silhouette Scores ---
        self._plot_silhouette(num_df, "Number")
        self._plot_silhouette(vol_df, "Volume Fraction")

        # --- Plot 3 & 4: RMSSTD (Elbow Method) ---
        self._plot_rmsstd(num_df, "Number")
        self._plot_rmsstd(vol_df, "Volume Fraction")

        print(f"[OK] All visualizations saved to: {self.output_dir}")

    # ----------------------------------------------------------------
    # --- Internal plotting functions ---
    # ----------------------------------------------------------------
    def _plot_silhouette(self, df, label):
        font = "Times New Roman"
        plt.rcParams["font.family"] = font
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        })
        if df.empty:
            print(f"[WARN] No Silhouette data for {label}")
            return

        plt.figure(figsize=(6, 4))
        plt.bar(df["Clusters"], df["Silhouette"], color="#006F98", edgecolor="black")

        # Add mean line
        avg_score = df["Silhouette"].mean()
        plt.axhline(y=avg_score, color="#5E0B4F", linestyle="--", linewidth=1.2) #,label=f"Mean = {avg_score:.3f}"
        
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Mean Silhouette Score")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / f"Silhouette_{label.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved {save_path.name}")

    def _plot_rmsstd(self, df, label):
        font = "Times New Roman"
        plt.rcParams["font.family"] = font
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        })

        if df.empty:
            print(f"[WARN] No RMSSTD data for {label}")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(df["Clusters"], df["RMSSTD"], marker="o", color="#006F98", linewidth=2)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Mean RMSSTD")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / f"RMSSTD_{label.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved {save_path.name}")

