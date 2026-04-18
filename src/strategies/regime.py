from src.core.registry import StrategyRegistry
from src.strategies.base import BaseStrategy
import numpy as np


@StrategyRegistry.register("regime")
class RegimeStrategy(BaseStrategy):

    def generate_weights(self, returns, cov):

        scores = returns.mean()

        w = np.maximum(scores.values, 0)

        if w.sum() == 0:
            return np.ones(len(w)) / len(w)

        return w / w.sum()
