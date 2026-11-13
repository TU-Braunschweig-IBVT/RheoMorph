from pathlib import Path
from Two.ClusterProcessor import PSDClusterProcessor
import tkinter as tk
from tkinter import filedialog

class MainApp:
    def __init__(self, base_path):
        self.base_path = base_path
        self.run()

    def run(self):
        processor = PSDClusterProcessor(self.base_path)
        processor.process()

if __name__ == "__main__":
    base_path=filedialog.askdirectory(title="Select folder with subfolders that contain fused morphology data (Clustering folder).")

    app = MainApp(base_path)
