from pathlib import Path
from Four.FinalEvaluation import MatrixEvaluator
import tkinter as tk
from tkinter import filedialog

class MainApp:
    def __init__(self, base_path, cluster_type, cluster_count):
        self.base_path = base_path
        self.cluster_type = cluster_type
        self.cluster_count = cluster_count
        self.run()

    def run(self):
        processor = MatrixEvaluator(self.base_path, self.cluster_type, self.cluster_count)
        processor.process()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    #base_path = current_dir / "Output"
    base_path=Path(filedialog.askdirectory(title="Select folder with subfolders that contain fused morphology data (Clustering folder)."))
    
    # --- Choose clustering mode manually ---
    cluster_type = "Volume"        # or "Number"
    cluster_count = 2              # e.g. 3 clusters

    app = MainApp(base_path, cluster_type, cluster_count)
