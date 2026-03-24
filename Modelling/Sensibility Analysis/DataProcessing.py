import numpy as np

class Processor:
    def __init__(self):
        self.cluster_count = 3
        self.metrics_per_cluster = 3
        self.max_values = None  # store max of each metric column
        self.metric_indices = []  # indices of metric columns (excluding VF)

    # -------------Column-wise Normalization-----------------
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        # identify metric columns (exclude VF)
        self.metric_indices = [i for i in range(X.shape[1])
                               if (i + 1) % (self.metrics_per_cluster + 1) != 0]

        # store max of each metric column
        self.max_values = []
        for idx in self.metric_indices:
            col_max = np.max(X[:, idx])
            # avoid division by zero
            if col_max == 0:
                col_max = 1.0
            self.max_values.append(col_max)

        return self

    def transform(self, X):
        X_out = X.copy().astype(float)
        for idx, col_max in zip(self.metric_indices, self.max_values):
            X_out[:, idx] = X_out[:, idx] / col_max
        return X_out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    # -------------Weigh by Volume Fraction-----------------
    def weight_by_vf_vectorized(self, X):
        X = np.asarray(X)
        X_weighted = []

        for row in X:
            arr = row.reshape(self.cluster_count, self.metrics_per_cluster + 1)
            metrics = arr[:, :self.metrics_per_cluster]
            vf = arr[:, self.metrics_per_cluster].reshape(self.cluster_count, 1)
            weighted_metrics = metrics * vf
            X_weighted.append(weighted_metrics.flatten())

        return np.vstack(X_weighted)

    # -------------Manual weights-----------------
    def apply_manual_weights(self, X, weights):
        w = np.array([
            weights["diameter"],
            weights["circularity"],
            weights["compactness"],
        ])
        X_out = X.copy().astype(float)
        n_clusters = self.cluster_count
        for c in range(n_clusters):
            start = c * self.metrics_per_cluster
            end = start + self.metrics_per_cluster
            X_out[:, start:end] = X_out[:, start:end] * w  # broadcasting
        return X_out
    
    # ---------------- Column-wise Normalization ----------------
    def fit_rheo(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.max_values = np.max(X, axis=0)
        # avoid division by zero
        self.max_values[self.max_values == 0] = 1.0
        return self

    def transform_rheo(self, X):
        X_out = np.asarray(X, dtype=float)
        X_out = X_out / self.max_values  # column-wise normalization
        return X_out

    def fit_transform_rheo(self, X, y=None):
        self.fit_rheo(X, y)
        return self.transform_rheo(X)
    
    # ---------------- Manual weights ----------------
    def apply_manual_weights_rheo(self, X, weights):
        """
        weights: dict, e.g. {"K": 0.5, "n": 1.0} 
        multiplies first feature by "K" and second by "n"
        """
        w = np.array([weights["K"], weights["n"]])
        X_out = np.asarray(X, dtype=float)
        X_out = X_out * w  # broadcasting
        return X_out
