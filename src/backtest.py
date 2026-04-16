import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio import portfolio_return, portfolio_volatility, negative_sharpe


def optimize_portfolio(mean_returns, cov_matrix, bounds, constraints, init_guess):
    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def run_backtest(prices, window=60):
    returns = prices.pct_change().dropna()

    tickers = prices.columns
    num_assets = len(tickers)

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init_guess = np.array([1 / num_assets] * num_assets)

    portfolio_returns = []

    for i in range(window, len(returns) - 1):
        train_data = returns.iloc[i - window : i]

        mean_returns = train_data.mean()
        cov_matrix = train_data.cov()

        weights = optimize_portfolio(
            mean_returns, cov_matrix, bounds, constraints, init_guess
        )

        next_return = np.dot(weights, returns.iloc[i + 1])

        portfolio_returns.append(next_return)

    return pd.Series(portfolio_returns)
