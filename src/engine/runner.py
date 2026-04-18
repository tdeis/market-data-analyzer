from src.core.registry import StrategyRegistry
from src.engine.pipeline import compute_returns
from src.core.experiment import ExperimentResult
from src.analytics.reporter import ResearchReporter
import numpy as np


class StrategyRunner:

    def __init__(self, prices, window=60, rebalance=21):

        self.prices = prices
        self.window = window
        self.rebalance = rebalance

    def run_strategy(self, name):

        StrategyClass = StrategyRegistry.get(name)
        strategy = StrategyClass()

        returns_df = compute_returns(self.prices)

        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        portfolio_returns = returns_df.dot(weights)

        metrics = ResearchReporter(portfolio_returns).summary()

        return ExperimentResult(
            strategy=name,
            returns=portfolio_returns,
            metrics=metrics,
            metadata={"window": self.window},
        )

    def run_all(self, names):

        results = []

        for n in names:
            results.append(self.run_strategy(n))  # MUST be ExperimentResult

        return results
