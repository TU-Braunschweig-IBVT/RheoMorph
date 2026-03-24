import numpy as np

def compute_averages_and_deviations(values):
    if not values:
        print("⚠️ No valid values to compute statistics.")
        return None, None
    avg = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    return avg, std