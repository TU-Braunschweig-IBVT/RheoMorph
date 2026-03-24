import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from pathlib import Path

class PowerLawFitter:
    def __init__(self):
        font = "Times New Roman"
        plt.rcParams["font.family"] = font
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams.update({
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        })

    @staticmethod
    def power_law(gamma, K, n):
        """Power law model: eta = K * gamma^n"""
        return K * gamma**n

    def fit(self, eta, gamma, tau, plot_id=None):
        """
        Simple linearised power law fit (log–log regression)
        Only removes points with gamma > 40 (tail)
        """
        eta = np.array(eta)
        gamma = np.array(gamma)
        tau = np.array(tau)

        # Keep only valid points
        mask = (tau > 5e-8) & (gamma >= 0.7) & np.isfinite(eta) & np.isfinite(gamma)
        eta = eta[mask]
        gamma = gamma[mask]

        # Keep only positive values
        mask = (eta > 0) & (gamma > 0)
        eta = eta[mask]
        gamma = gamma[mask]
        
        # Remove tail (gamma > 40)
        mask = gamma <= 40
        eta = eta[mask]
        gamma = gamma[mask]

        # Safety check
        if len(gamma) < 2:
            print("[WARN] Not enough points for fit")
            return np.nan, np.nan, np.nan

        # Linearised fit
        x = np.log(gamma)
        y = np.log(eta)
        p = np.polyfit(x, y, 1)
        n = p[0] + 1
        lnK = p[1]
        K = np.exp(lnK)

        y_pred = np.polyval(p, x)
        r2 = r2_score(y, y_pred)

        # Plot
        pid = plot_id if plot_id else f"linearised_{np.random.randint(1e6)}"
        self._plot_linearised_save(gamma, eta, K, n, r2, pid)

        return K, n, r2

    def _plot_linearised_save(self, gamma, eta, K, n, r2, plot_id):
    #def _plot_linearised_save(self, gamma_fit, eta_fit, gamma_excl, eta_excl, K, n, r2, plot_id): 
        current_dir = Path(__file__).parent
        outdir = current_dir / "Evaluate" / "Plots"
        os.makedirs(outdir, exist_ok=True)

        plt.figure(figsize=(6, 4))
        plt.scatter(gamma, eta, color="black") #, label="Data"

        gamma_fit = np.logspace(np.log10(min(gamma)), np.log10(max(gamma)), 200)
        eta_fit = self.power_law(gamma_fit, K, (n-1))

        plt.plot(gamma_fit, eta_fit, color="#5E0B4F",
                 label=f"$\\mu_{{\\mathrm{{app}}}} = {K:.3f}·\\gamma^{{{(n-1):.2f}}}$\nR²={r2:.3f}")

        #-----------------------------
        plt.xlabel("Shear rate γ [$\mathrm{s}^{-1}$]")
        plt.ylabel("apparent viscosity $\mu_{\mathrm{app}}$ [Pa·s]")
        plt.xscale("log")
        plt.yscale("log")
        xmin_decade = 10 ** np.floor(np.log10(min(gamma)))
        xmax_decade = 10 ** np.ceil(np.log10(max(gamma)))
        ymin_decade = 10 ** np.floor(np.log10(min(eta)))
        ymax_decade = 10 ** np.ceil(np.log10(max(eta)))
        plt.xlim(xmin_decade, xmax_decade)
        plt.ylim(ymin_decade, ymax_decade)

        plt.legend()
        #plt.grid(True, ls="--", alpha=0.5, which="both")
        plt.tight_layout()

        fname = f"{plot_id}.png"
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()