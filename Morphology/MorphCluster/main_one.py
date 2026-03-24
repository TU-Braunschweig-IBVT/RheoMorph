from pathlib import Path
from One.FolderSearcher import FolderProcessor

current_dir = Path(__file__).parent

class MainApp:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        self.output_path = output_path
        self.run()

    def run(self):
        print(f"[INFO] Starting folder consolidation in {self.base_path}")
        processor = FolderProcessor(self.base_path, self.output_path)
        processor.process()

if __name__ == "__main__":
    base_path = Path(__file__).parent / "Input"
    output_path = Path(__file__).parent / "Output"

    app = MainApp(base_path, output_path)
