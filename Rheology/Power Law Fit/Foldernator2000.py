import os
import pandas as pd
from pathlib import Path
from PowerLawFitter import PowerLawFitter 

class FolderScanner:
    def __init__(self):
        self.fitter = PowerLawFitter()

    def _find_columns_with_header_guess(self, file_path, required_keys=('eta','gamma','tau'), max_header_row=5):
        """
        Tries header rows 0..max_header_row to find one where the dataframe's columns
        contain substrings for all required_keys. Returns (df, mapping, header_row) or (None, None, None).
        mapping: dict like {'sigma': actual_col_name_in_df, ...}
        """
        for h in range(max_header_row + 1):
            try:
                df = pd.read_excel(file_path, header=h)
            except Exception:
                continue

            cols = list(df.columns)
            # normalized column names for substring matching
            norm = []
            for c in cols:
                if isinstance(c, str):
                    norm.append(c.strip().lower())
                else:
                    norm.append(str(c).lower())

            mapping = {}
            for key in required_keys:
                found = None
                for i, c_norm in enumerate(norm):
                    if key in c_norm:  # substring match
                        found = cols[i]
                        break
                if found:
                    mapping[key] = found
                else:
                    break

            if len(mapping) == len(required_keys):
                return df, mapping, h

        return None, None, None

    def scan(self, top_folder):
        """
        Scan through top_folder and its subfolders for Excel files,
        extract sigma/gamma/tau arrays, fit them using PowerLawFitter,
        and return a results dictionary.
        """
        results = {}

        for root, dirs, files in os.walk(top_folder):
            for file in files:
                if not file.lower().endswith((".xlsx", ".xls")):
                    continue

                file_path = Path(root) / file
                sample_name = file_path.stem

                try:
                    # Try to detect the header row and the correct column names
                    df, mapping, header_row = self._find_columns_with_header_guess(file_path)
                    if df is None:
                        print(f"⚠️ Could not find required columns in {file_path}. Tried multiple header rows.")
                        results[sample_name] = {"K": None, "n": None, "R²": None}
                        continue

                    # Convert columns to numeric, coerce errors -> NaN
                    eta   = pd.to_numeric(df[mapping['eta']], errors='coerce').to_numpy()
                    gamma = pd.to_numeric(df[mapping['gamma']], errors='coerce').to_numpy()
                    tau   = pd.to_numeric(df[mapping['tau']], errors='coerce').to_numpy()



                    # Optional: drop rows where any of the needed values are NaN
                    valid_mask = (~pd.isna(eta)) & (~pd.isna(gamma)) & (~pd.isna(tau))
                    if valid_mask.sum() < 2:
                        print(f"⚠️ Not enough numeric rows in {file_path} after coercion. Found {valid_mask.sum()} valid rows.")
                        results[sample_name] = {"K": None, "n": None, "R²": None}
                        continue

                    eta = eta[valid_mask]
                    gamma = gamma[valid_mask]
                    tau   = tau[valid_mask]

                    # Call the fitter (expecting fit(sigma, gamma, tau) -> (K, n, r2) or None)
                    fit_result = self.fitter.fit(eta, gamma, tau, sample_name)

                    if fit_result:
                        K, n, r2 = fit_result
                        results[sample_name] = {"K": K, "n": n, "R²": r2}
                    else:
                        results[sample_name] = {"K": None, "n": None, "R²": None}

                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    results[sample_name] = {"K": None, "n": None, "R²": None}

        return results
