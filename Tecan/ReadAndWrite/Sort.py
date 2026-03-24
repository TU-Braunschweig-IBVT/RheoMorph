import pandas as pd
from pathlib import Path


class SortResult:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def sort_file(self):
        """
        Sorts the Excel file by extracting the numeric part of sample names like 'DOE 2.3'.
        """
        if not self.file_path.exists():
            print("❌ File not found for sorting.")
            return

        df = pd.read_excel(self.file_path)

        # Extract numeric sort key from 'Sample' assuming format like 'DOE 2.3'
        def extract_sort_key(name):
            try:
                return float(name.split()[1])
            except Exception:
                return float('inf')  # Push malformed entries to end

        df["_sort"] = df["Sample"].apply(extract_sort_key)
        df = df.sort_values(by="_sort")
        df.drop(columns=["_sort"], inplace=True)
        df.to_excel(self.file_path, index=False)
        print(f"✅ File sorted numerically by Sample at {self.file_path}")