from Foldernator2000 import FolderScanner
from Writer import ResultWriter
from pathlib import Path


current_dir = Path(__file__).parent

input_folder = current_dir / "Evaluate" / "Input"
output_file = current_dir / "Evaluate" / "Output" / "Data collection.xlsx"


class main:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.scanner = FolderScanner()

    def run(self):
        print(f"🔍 Scanning folder: {self.folder_path}")
        results = self.scanner.scan(self.folder_path)

        # Test output
        print("\n📊 Scan Results:")
        for sample, vals in results.items():
            print(f"{sample}: K={vals['K']}, n={vals['n']}, R²={vals['R²']}")
        
        # Write results to Excel file
        writer = ResultWriter(output_file)
        writer.write(results)

        return results


if __name__ == "__main__":
    app = main(input_folder)
    app.run()
