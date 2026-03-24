# file: synthetic_data.py
import numpy as np

class SyntheticDataGenerator:
    """
    Generates synthetic morphology vectors and concentration values.
    """

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def expand_morphology(self, means, stds, volume_fractions):
        means = np.asarray(means)
        stds = np.asarray(stds)
        vf = np.asarray(volume_fractions)

        # Replace zero std with 30% of the mean
        safe_stds = stds.copy()
        zero_mask = safe_stds <= 0

        safe_stds[zero_mask] = np.abs(means[zero_mask] * 0.3)

        # Generate synthetic metric samples
        syn_metrics = self.rng.normal(
            loc=means,
            scale=safe_stds,
            size=(self.n, means.shape[0], means.shape[1])
        )

        syn_metrics = np.clip(syn_metrics, a_min=0, a_max=None)

        # Reassemble with fixed volume fractions
        synthetic_vectors = []
        for i in range(self.n):
            combined = np.column_stack([syn_metrics[i], vf])
            synthetic_vectors.append(combined.flatten())

        return np.array(synthetic_vectors)

    def expand_concentration(self, mean_value, std_value):
        # NEW: Same rule for concentrations — optional but consistent
        if std_value <= 0:
            std_value = abs(mean_value) * 0.3
            if mean_value == 0:
                std_value = 0.001

        return self.rng.normal(loc=mean_value, scale=std_value, size=self.n)
