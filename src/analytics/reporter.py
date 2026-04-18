import numpy as np


class ResearchReporter:

    def __init__(self, returns):
        self.returns = returns

    def summary(self):

        cum = self.returns.cumsum()
        peak = cum.cummax()
        drawdown = (peak - cum).max()

        return {
            "mean": float(self.returns.mean()),
            "volatility": float(self.returns.std()),
            "sharpe": float(self.returns.mean() / (self.returns.std() + 1e-8)),
            "drawdown": float(drawdown),  # ✅ NOW ALWAYS EXISTS
        }
