import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
from pathlib import Path
import numpy as np
import random

class PSDPlotter:
    def __init__(self, file_path: Path, cluster_type: str, cluster_count: int, bins: int = 25):
        self.file_path = Path(file_path)
        self.cluster_type = cluster_type
        self.k = cluster_count
        self.bins = bins
        self.df = None

    def load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        skip = 1 if first_line.startswith("#") else 0

        self.df = pd.read_csv(self.file_path, skiprows=skip)
        self.df.columns = self.df.columns.str.strip()
        print(f"[INFO] Loaded data from {self.file_path.name}")

    def plot_psd(self):
        if self.df is None:
            self.load_data()
        font = "Times New Roman"
        plt.rcParams["font.family"] = font
        plt.rcParams.update({
                    "font.family": "Times New Roman",
                    "font.size": 20,
                    "axes.titlesize": 20,
                    "axes.labelsize": 20,
                    "xtick.labelsize": 20,
                    "ytick.labelsize": 20,
                    "legend.fontsize": 20,
                })

        col_diam = "Diameter [mym]"
        col_volfrac = "Volume Fraction"
        cluster_col = f"Cluster_{self.cluster_type}_k{self.k}"

        df_sorted = self.df.sort_values(by=col_diam)
        diam = df_sorted[col_diam].values
        volfrac = df_sorted[col_volfrac].values
        cluster_labels = df_sorted[cluster_col].values

        # --- Determine cluster boundaries ---
        cluster_boundaries = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i - 1]:
                border = (diam[i] + diam[i - 1]) / 2
                cluster_boundaries.append(border)

        # --- Compute binned volume-weighted size distribution ---
        hist, edges = np.histogram(
            diam, bins=self.bins, weights=volfrac, density=False
        )
        hist = hist / np.sum(hist)

        plt.figure(figsize=(8, 5))

        # ----------------------------------------
        # 1) Soft tail: add one extra bin with tiny epsilon
        # ----------------------------------------
        bin_width = edges[1] - edges[0]

        # histogram centers
        centers = 0.5 * (edges[1:] + edges[:-1])

        # extend centers with one additional point
        extra_center = centers[-1] + bin_width
        centers_ext = np.append(centers, extra_center)

        # extend histogram with VERY small positive tail
        epsilon = hist.max() * 0.02   # 2% of max value → natural decay
        hist_ext = np.append(hist, epsilon)

        # ----------------------------------------
        # 2) Plot bars (same as before)
        # ----------------------------------------
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            align="edge",
            edgecolor="black",
            linewidth=0.5,
            color="white"
        )

        # ----------------------------------------
        # 3) Smooth curve
        # ----------------------------------------
        spline = make_interp_spline(centers_ext, hist_ext, k=3)

        x_smooth = np.linspace(centers_ext.min(), centers_ext.max(), 800)
        y_smooth = spline(x_smooth)

        # Clip negative dips
        y_smooth = np.clip(y_smooth, 0, None)

        #plt.plot(x_smooth, y_smooth, color="#5E0B4F", lw=1.8)

        # cluster borders unchanged
        for border in cluster_boundaries:
            plt.axvline(x=border, color="#006F98", linestyle="--", lw=2)

        plt.xlabel("Diameter [μm]")
        plt.ylabel("Volume Fraction [1]")
        plt.tight_layout()


        out_path = self.file_path.parent / f"PSD_{self.cluster_type}_k{self.k}_Distribution.png"
        plt.savefig(out_path, dpi=300)
        if random.random() < 0.01:
            plt.show()
        plt.close()
        print(f"[OK] Saved: {out_path.name}")


        # --- Plot Cumulative PSD ---
        cum_hist = np.cumsum(hist)
        # --- Plot Cumulative PSD ---
        plt.figure(figsize=(8, 5))

        # ----------------------------------------
        # 1) EXTEND BINS WITH 2 EMPTY BINS
        # ----------------------------------------
        extra_bins = 2
        bin_width = edges[1] - edges[0]

        extra_edges = edges[-1] + np.arange(1, extra_bins + 1) * bin_width
        edges_ext = np.concatenate([edges, extra_edges])

        hist_ext = np.concatenate([hist, np.zeros(extra_bins)])

        # ----------------------------------------
        # 2) CUMULATIVE (ensure never below zero)
        # ----------------------------------------
        cum_hist = np.cumsum(hist_ext)
        cum_hist = cum_hist / cum_hist[-1]   # normalize
        cum_hist = np.clip(cum_hist, 0, None)  # prevent small negative dips

        # ----------------------------------------
        # 3) Plot bars
        # ----------------------------------------
        bin_widths_ext = edges_ext[1:] - edges_ext[:-1]

        plt.bar(
            edges_ext[:-1],
            cum_hist,
            width=bin_widths_ext,
            align="edge",
            edgecolor="black",
            linewidth=0.5,
            color="white"
        )

        # ----------------------------------------
        # 4) Smooth curve
        # ----------------------------------------
        centers_ext = 0.5 * (edges_ext[1:] + edges_ext[:-1])

        spline = make_interp_spline(centers_ext, cum_hist)
        x_smooth = np.linspace(centers_ext.min(), centers_ext.max(), 800)
        y_smooth = spline(x_smooth)

        #plt.plot(x_smooth, y_smooth, color="#5E0B4F", lw=1.8)

        # ----------------------------------------
        # 5) Cluster borders again
        # ----------------------------------------
        for border in cluster_boundaries:
            plt.axvline(x=border, color="#006F98", linestyle="--", lw=2)

        plt.xlabel("Diameter [μm]")
        plt.ylabel("Cumulative Volume Fraction [1]")
        plt.tight_layout()
        
        out_path = self.file_path.parent / f"PSD_{self.cluster_type}_k{self.k}_Cumulative.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] Saved: {out_path.name}")


    def run(self):
        self.load_data()
        self.plot_psd()
