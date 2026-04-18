from src.core.loader import load_all_strategies
from src.engine.runner import StrategyRunner
from src.experiments.leaderboard import build_leaderboard
from src.db.database import ExperimentDB


def run_research(tickers):

    import pandas as pd

    prices = pd.DataFrame()  # replace with loader later

    load_all_strategies()

    from src.core.registry import StrategyRegistry

    names = list(StrategyRegistry._registry.keys())

    runner = StrategyRunner(prices)

    results = runner.run_all(names)

    leaderboard = build_leaderboard(results)

    db = ExperimentDB()

    for r in results:
        db.log(r.strategy, r.metrics, r.metadata)

    return {
        "results": results,
        "leaderboard": leaderboard,
        "db_leaderboard": db.leaderboard(),
    }
