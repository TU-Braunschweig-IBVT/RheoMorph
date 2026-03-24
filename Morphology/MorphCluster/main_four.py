from pathlib import Path
from Four.FinalEvaluation import MatrixEvaluator

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
    base_path = current_dir / "Output"

    # --- Choose clustering mode manually ---
    cluster_type = "Volume"        # or "Number"
    cluster_count = 3              # e.g. 3 clusters

    app = MainApp(base_path, cluster_type, cluster_count)
