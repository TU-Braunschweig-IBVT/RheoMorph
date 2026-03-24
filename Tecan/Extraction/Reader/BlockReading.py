import pandas as pd

class BlockReader:
    def __init__(self, dataframe, marker_index, sample_methods, sample_names, sample_positions, blank, method):
        self.blank = blank.lower()
        self.method = method.lower()
        self.df = dataframe
        self.idx = marker_index
        self.sample_methods = sample_methods
        self.sample_names = sample_names
        self.sample_positions = sample_positions
        self.block = None
        self.blank_value = None
        self.sample_data = {}

    def process_block(self):
        try:
            self._extract_block()
            self._find_blank()
            self._extract_corrected_sample_values()
        except Exception as e:
            raise RuntimeError(f"BlockReader failed at marker index {self.idx}: {e}")
        return self.sample_data

    def _extract_block(self):
        try:
            self.block = self.df.iloc[self.idx+1:self.idx+9, :13]
            self.block.columns = ['Row'] + list(range(1, 13))
            self.block.set_index('Row', inplace=True)
        except Exception as e:
            raise ValueError(f"Could not extract or format block starting at index {self.idx}: {e}")

    def _find_blank(self):
        for i, method in enumerate(self.sample_methods):
            if method.lower() == self.blank:
                row, col = self.sample_positions[i]
                val = self._get_value(row, col)
                if val is not None:
                    self.blank_value = val
                    #print(f"✅ Found blank value {val} at position ({row}, {col})")
                    return
        raise ValueError(f"No valid blank '{self.blank}' found in block at index {self.idx}")

    def _extract_corrected_sample_values(self):
        for i, method in enumerate(self.sample_methods):
            if method.lower() == self.method:
                row, col = self.sample_positions[i]
                name = self.sample_names[i]
                raw_val = self._get_value(row, col)
                if raw_val is not None:
                    corrected = raw_val - self.blank_value
                    self.sample_data.setdefault(name, []).append(corrected)
                    #print(f"🔹 {name}: raw={raw_val}, corrected={corrected}")
                else:
                    print(f"⚠️ Missing value for sample '{name}' at ({row}, {col})")

    def _get_value(self, row, col):
        try:
            val = self.block.at[row, col]
            return float(val) if pd.notnull(val) else None
        except Exception as e:
            print(f"⚠️ Failed to read value at ({row}, {col}): {e}")
            return None
