from pathlib import Path
from Three.ClusterValidator import ClusterValidator
from Three.Visualisation import ClusterVisualizer
import tkinter as tk
from tkinter import filedialog

class MainApp:
    def __init__(self, base_path):
        self.base_path = base_path
        self.run()

    def run(self):
        processor = ClusterValidator(self.base_path)
        processor.process()
    
        # Visualize results
        eval_folder = self.base_path / "Evaluation Clustering"
        visualizer = ClusterVisualizer(eval_folder)
        visualizer.visualize()

if __name__ == "__main__":
    #current_dir = Path(__file__).parent
    #base_path = current_dir / "Output"
    base_path=Path(filedialog.askdirectory(title="Select folder with subfolders that contain fused morphology data (Clustering folder)."))
    
    app = MainApp(base_path)
