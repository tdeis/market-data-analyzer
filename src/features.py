import numpy as np
import pandas as pd


# =========================
# VALIDATION
# =========================
def _validate_prices(prices):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices is empty")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")

    if prices.isnull().all().any():
        raise ValueError("One or more assets contain only NaNs")


# =========================
# CORE RETURNS
# =========================
def compute_returns(prices, method="pct"):

    _validate_prices(prices)

    if method == "pct":
        returns = prices.pct_change()
    else:
        raise ValueError("Unsupported return method")

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    return returns


# =========================
# ROLLING VOLATILITY
# =========================
def rolling_volatility(returns, window=20):

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")

    if len(returns) < window:
        raise ValueError("Not enough data for rolling volatility")

    return returns.rolling(window).std() * np.sqrt(252)


# =========================
# MOMENTUM FEATURES
# =========================
def momentum(returns, window=20):

    if len(returns) < window:
        raise ValueError("Not enough data for momentum")

    return returns.rolling(window).mean()


# =========================
# CUMULATIVE RETURN (EQUITY)
# =========================
def cumulative_returns(returns):

    if returns.empty:
        raise ValueError("returns is empty")

    return (1 + returns).cumprod()


# =========================
# DRAWDOWN
# =========================
def drawdown(returns):

    equity = cumulative_returns(returns)

    peak = equity.cummax()
    dd = (equity - peak) / peak

    return dd


# =========================
# CORRELATION MATRIX
# =========================
def rolling_correlation(returns, window=60):

    if len(returns) < window:
        raise ValueError("Not enough data for rolling correlation")

    return returns.rolling(window).corr()


# =========================
# FEATURE PIPELINE WRAPPER
# =========================
def build_feature_set(prices):

    _validate_prices(prices)

    returns = compute_returns(prices)

    features = {}

    # returns
    features["returns"] = returns

    # volatility
    features["volatility"] = rolling_volatility(returns)

    # momentum
    features["momentum"] = momentum(returns)

    # drawdown (portfolio-level proxy)
    features["drawdown"] = drawdown(returns.mean(axis=1))

    # cumulative performance
    features["equity"] = cumulative_returns(returns.mean(axis=1))

    return features
