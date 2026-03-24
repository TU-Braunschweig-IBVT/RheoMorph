import os
from pathlib import Path
import pandas as pd
import numpy as np

class FolderProcessor:
    def __init__(self, base_path: Path, output_base: Path):
        self.base_path = Path(base_path)
        self.output_base = Path(output_base)

    def process(self):
        print(f"[INFO] Processing base path: {self.base_path}")

        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            csv_files = [f for f in files if f.endswith(".csv") or f.endswith(".xlsx")]

            if not csv_files:
                continue

            print(f"[DBG] Found {len(csv_files)} files in {root_path.relative_to(self.base_path)}")

            frames = []
            for file_name in csv_files:
                file_path = root_path / file_name
                if file_name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                frames.append(df)

            combined = pd.concat(frames, ignore_index=True)

            # 🔧 FIX: invert wrongly calculated columns
            columns_to_invert = [
                "Irregularity [-]",
                "Diameter Ratio [-]"
            ]

            for col in columns_to_invert:
                if col in combined.columns:
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")
                    combined[col] = 1 / combined[col]
                else:
                    print(f"[WARN] Column '{col}' not found in {root_path}")

            # Create mirrored output path
            relative_path = root_path.relative_to(self.base_path)
            output_dir = self.output_base / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "combined_data.csv"
            combined.to_csv(output_file, index=False)

            print(f"[INFO] Saved combined file to: {output_file}")

        print("[DONE] Folder processing finished.")
