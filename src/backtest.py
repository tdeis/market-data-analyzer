import numpy as np
import pandas as pd

from src.optimizer import optimize_portfolio
from src.portfolio import negative_sharpe, negative_sharpe_defensive
from src.regime import detect_regime


def _validate(prices):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be DataFrame")
    if prices.isnull().values.any():
        raise ValueError("NaNs in price data")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")


def run_backtest(
    prices, window=60, rebalance_freq=21, transaction_cost=0.001, max_weight=0.4
):

    _validate(prices)

    returns = prices.pct_change().dropna()

    if len(returns) <= window:
        raise ValueError("Not enough data")

    n = returns.shape[1]

    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    regime = detect_regime(prices)

    init = np.ones(n) / n
    prev_w = init.copy()

    r_vals = returns.values
    idx = returns.index

    out_returns = []
    out_dates = []
    weights_hist = []

    for i in range(window, len(r_vals) - 1):

        train = r_vals[i - window : i]

        mu = train.mean(axis=0)
        cov = np.cov(train, rowvar=False)

        obj = negative_sharpe if regime.iloc[i] == "bull" else negative_sharpe_defensive

        if i % rebalance_freq == 0:
            w = optimize_portfolio(mu, cov, bounds, constraints, init, obj)
        else:
            w = prev_w

        turnover = np.sum(np.abs(w - prev_w))
        cost = transaction_cost * turnover

        ret = np.dot(w, r_vals[i + 1]) - cost

        out_returns.append(ret)
        out_dates.append(idx[i + 1])
        weights_hist.append(w)

        prev_w = w

    return (
        pd.Series(out_returns, index=out_dates, name="strategy"),
        pd.DataFrame(weights_hist, index=out_dates, columns=prices.columns),
    )
