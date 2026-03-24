import pandas as pd
import numpy as np
from collections import defaultdict

class FluorescenceProcessor:
    def __init__(self, dataframe, sample_methods, sample_names, sample_dilutions):
        self.sample_methods = sample_methods
        self.sample_names = sample_names
        self.sample_dilutions = sample_dilutions
        self.df = dataframe
        self.values_set_1 = []
        self.values_set_2 = []
        self.result_by_sample = defaultdict(list)

    def _extract_setup(self):
        self.find_544_values()
        self.apply_blank_correction()
        full_data = self.compute_averages()
        return full_data

    def find_544_values(self):
        matches = self.df[self.df.iloc[:, 0].astype(str) == "544"]
        if len(matches) < 1:
            raise ValueError("No row starting with '544' found.")
        elif len(matches) == 1:
            print("⚠️ Only one '544' row found. Fluorescence will be calculated with one replicate.")
            idx1 = matches.index[0]
            self.values_set_1 = self.df.iloc[idx1, 1:].tolist()
            self.values_set_2 = [None] * len(self.values_set_1)
        else:
            idx1, idx2 = matches.index[:2]
            self.values_set_1 = self.df.iloc[idx1, 1:].tolist()
            self.values_set_2 = self.df.iloc[idx2, 1:].tolist()

    def apply_blank_correction(self):
        blank_1, blank_2 = 0, 0

        for i, method in enumerate(self.sample_methods):
            if method == "Blank":
                v1 = self._safe_float(self.values_set_1[i])
                v2 = self._safe_float(self.values_set_2[i])
                if v1 is not None: blank_1 = v1
                if v2 is not None: blank_2 = v2

        for i, method in enumerate(self.sample_methods):
            if method == "f":
                dilution = self._safe_float(self.sample_dilutions[i])
                name = self.sample_names[i]

                v1 = self._safe_float(self.values_set_1[i])
                if v1 is not None:
                    corrected_v1 = v1 * dilution - blank_1
                    self.result_by_sample[name].append(corrected_v1)

                v2 = self._safe_float(self.values_set_2[i])
                if v2 is not None:
                    corrected_v2 = v2 * dilution - blank_2
                    self.result_by_sample[name].append(corrected_v2)

    def compute_averages(self):
        excluded_names = {"Blank", "BlankB", "BlankR"}
        unique_names = sorted({name for name in self.sample_names if name not in excluded_names})

        full_data = {}

        for name in unique_names:
            values = self.result_by_sample.get(name, [])
            clean_vals = [v for v in values if v is not None]

            if clean_vals:
                avg = np.mean(clean_vals)
                std = np.std(clean_vals, ddof=1) if len(clean_vals) > 1 else 0
            else:
                avg = None
                std = None

            full_data[name] = {
                "fluorescence_avg": avg,
                "fluorescence_std": std
            }

        return full_data

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
