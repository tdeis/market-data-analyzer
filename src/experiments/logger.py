import json
import pandas as pd
from pathlib import Path
from datetime import datetime


class ExperimentLogger:

    def __init__(self, base_path="experiments"):

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.base_path / "runs.csv"

        if not self.index_file.exists():
            pd.DataFrame(
                columns=[
                    "run_id",
                    "timestamp",
                    "strategy",
                    "tickers",
                    "sharpe",
                    "max_drawdown",
                    "volatility",
                ]
            ).to_csv(self.index_file, index=False)

    # =========================
    # CREATE RUN FOLDER
    # =========================
    def create_run(self):

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        return run_id, run_path

    # =========================
    # SAVE METRICS
    # =========================
    def log_run(self, run_id, strategy, tickers, metrics):

        df = pd.read_csv(self.index_file)

        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "tickers": ",".join(tickers),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "volatility": metrics.get("volatility"),
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.to_csv(self.index_file, index=False)

    # =========================
    # SAVE JSON METRICS
    # =========================
    def save_metrics_json(self, run_path, metrics):

        with open(run_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    # =========================
    # GET LEADERBOARD
    # =========================
    def leaderboard(self, top_n=10):

        df = pd.read_csv(self.index_file)
        return df.sort_values(by="sharpe", ascending=False).head(top_n)
