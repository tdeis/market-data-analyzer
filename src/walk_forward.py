import numpy as np
import pandas as pd

from src.optimizer import optimize_portfolio
from src.portfolio import negative_sharpe


def walk_forward(prices, train_window=120, test_window=21, max_weight=0.4):

    returns = prices.pct_change().dropna()

    if len(returns) < train_window + test_window:
        raise ValueError("Not enough data for walk-forward")

    n_assets = returns.shape[1]

    bounds = tuple((0, max_weight) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    init_guess = np.ones(n_assets) / n_assets

    results = []
    dates = []

    i = train_window

    while i < len(returns) - test_window:

        train = returns.iloc[i - train_window : i]
        test = returns.iloc[i : i + test_window]

        weights = optimize_portfolio(
            train.mean(), train.cov(), bounds, constraints, init_guess, negative_sharpe
        )

        for t in range(len(test)):
            results.append(np.dot(weights, test.iloc[t]))
            dates.append(test.index[t])

        i += test_window

    return pd.Series(results, index=dates)
