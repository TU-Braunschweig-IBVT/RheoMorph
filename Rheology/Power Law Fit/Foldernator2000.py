import os
import pandas as pd
from pathlib import Path
from PowerLawFitter import PowerLawFitter 

class FolderScanner:
    def __init__(self):
        self.fitter = PowerLawFitter()

    # Expected column layout, used when a file has no header row.
    # Only the positions of gamma (6), eta (7) and tau (11) matter for the fit;
    # the last few names are approximate.
    DEFAULT_HEADER = [
        "Index", "Sample Name", "texp (s)", "t (s)", "T (°C)",
        "sigma (Pa)", "gamma (s^-1)", "eta (Pas)", "F(N)", "N1 (Pa)",
        "g (mm)", "tau (Nm)", "gsigma (rad)", "steady state",
        "sigmaabs (Pa)", "Note",
    ]

    @staticmethod
    def _map_columns(df, required_keys):
        """Return {key: actual_col_name} if all keys are found as substrings of column names, else None."""
        cols = list(df.columns)
        norm = [c.strip().lower() if isinstance(c, str) else str(c).lower() for c in cols]

        mapping = {}
        for key in required_keys:
            for i, c_norm in enumerate(norm):
                if key in c_norm:  # substring match
                    mapping[key] = cols[i]
                    break
            else:
                return None
        return mapping

    def _find_columns_with_header_guess(self, file_path, required_keys=('eta','gamma','tau'), max_header_row=5):
        """
        Tries header rows 0..max_header_row to find one where the dataframe's columns
        contain substrings for all required_keys. If none is found, the file is assumed
        to have no header and DEFAULT_HEADER is applied by column position.
        Returns (df, mapping, header_row) or (None, None, None).
        header_row is None when the default header was used.
        mapping: dict like {'eta': actual_col_name_in_df, ...}
        """
        for h in range(max_header_row + 1):
            try:
                df = pd.read_excel(file_path, header=h)
            except Exception:
                continue

            mapping = self._map_columns(df, required_keys)
            if mapping:
                return df, mapping, h

        # Fallback: no header row -> read all rows as data and apply the default header
        try:
            df = pd.read_excel(file_path, header=None)
        except Exception:
            return None, None, None

        ncols = df.shape[1]
        header = list(self.DEFAULT_HEADER)
        if ncols == len(header) + 1:
            # Headerless exports contain an extra text column (e.g. "Viscometry shear
            # rate table") right after the sample name.
            header.insert(2, "Test name")
        df.columns = header[:ncols] + [f"col_{i}" for i in range(len(header), ncols)]
        mapping = self._map_columns(df, required_keys)
        if mapping:
            print(f"ℹ️ No header found in {file_path}, using default headers.")
            return df, mapping, None

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
