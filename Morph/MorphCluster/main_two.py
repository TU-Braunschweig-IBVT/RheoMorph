from pathlib import Path
from Two.ClusterProcessor import PSDClusterProcessor

class MainApp:
    def __init__(self, base_path):
        self.base_path = base_path
        self.run()

    def run(self):
        processor = PSDClusterProcessor(self.base_path)
        processor.process()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    base_path = current_dir / "Output"
    app = MainApp(base_path)
