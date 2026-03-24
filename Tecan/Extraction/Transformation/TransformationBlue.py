import pandas as pd
import numpy as np
from Extraction.Transformation.avg_std import compute_averages_and_deviations

class TransformationBlue:
    def __init__(self, values):
        self.values = values

    def transformation(self):
        transformed_values = self.calculation()
        avg_dev_blue = compute_averages_and_deviations(transformed_values)
        return avg_dev_blue

    def calculation(self):
        transformed = []
        for val in self.values:
            if isinstance(val, (int, float, np.number)) and not pd.isna(val):
                converted = val * 2 / 25320 / 0.56  # dilution / molar extinction / height → gives mol/L
                transformed.append(converted)
            else:
                print("⚠️ Non-numeric or missing value encountered, skipping transformation.")
        return transformed
