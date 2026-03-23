import numpy as np
from synthetic_data import SyntheticDataGenerator

class DataGeneration:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        

        
    def generate(self, X_mean, X_std, y, y_sd, sample_names):

        def determine_synthetic_count(mean, sd):
            try:
                mean = float(mean)
                sd = float(sd)
            except Exception:
                print("Error in determine synthetic count conversion.")

            r = sd / mean
            if 1/(r) <= 1:
                return 0
            elif 1/(r**2) < 20:
                return round(1/(r**2))
            else:
                return 20
        
        gen = SyntheticDataGenerator()
        X_aug = []
        y_aug = []
        sample_aug = []

        for i in range(len(X_mean)):
            # add real sample
            X_aug.append(X_mean[i])
            y_aug.append(y[i])
            sample_aug.append(sample_names[i])

            # dynamic synthetic count based on target uncertainty
            k_syn = determine_synthetic_count(y[i], y_sd[i])
            if k_syn <= 0:
                continue
            gen.n = k_syn

            # reshape means/stds
            row_mean = X_mean[i].reshape(3, 4)
            row_std = X_std[i].reshape(3, 4)

            means_matrix = row_mean[:, :3]
            stds_matrix = row_std[:, :3]
            vf_vector = row_mean[:, -1]

            syn_X = gen.expand_morphology(means_matrix, stds_matrix, vf_vector)
            syn_y = gen.expand_concentration(y[i], y_sd[i])

            X_aug.extend(syn_X)
            y_aug.extend(syn_y)
            for k in range(k_syn):
                sample_aug.append(f"{sample_names[i]}_syn{k}")

        X = np.array(X_aug)
        y = np.array(y_aug)
        sample_names = sample_aug
        
        return X, y, sample_names
    pass

    def generate_rheo(self, X_mean, X_std, y, y_sd, sample_names):
        """
        Generate synthetic samples based on rheology means and SDs.
        """
        def determine_synthetic_count(mean, sd):
            try:
                mean = float(mean)
                sd = float(sd)
            except Exception:
                print("Error in determine synthetic count conversion.")
            r = sd / mean
            if 1/(r) <= 1:
                return 0
            elif 1/(r**2) < 20:
                return round(1/(r**2))
            else:
                return 20
        
        X_aug = []
        y_aug = []
        sample_aug = []

        for i in range(len(X_mean)):
            # Add the real sample
            X_aug.append(X_mean[i])
            y_aug.append(y[i])
            sample_aug.append(sample_names[i])

            # Determine number of synthetic samples based on target uncertainty
            k_syn = determine_synthetic_count(y[i], y_sd[i])
            if k_syn <= 0:
                continue

            # Generate synthetic rheology
            means = X_mean[i]
            stds  = X_std[i]
            safe_stds = np.where(stds <= 0, np.abs(means*0.3), stds)
            syn_X = self.rng.normal(loc=means, scale=safe_stds, size=(k_syn, len(means)))

            # Generate synthetic targets
            safe_y_sd = y_sd[i] if y_sd[i] > 0 else abs(y[i])*0.3
            syn_y = self.rng.normal(loc=y[i], scale=safe_y_sd, size=k_syn)

            X_aug.extend(syn_X)
            y_aug.extend(syn_y)
            for k in range(k_syn):
                sample_aug.append(f"{sample_names[i]}_syn{k}")

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)
        return X_aug, y_aug, sample_aug
