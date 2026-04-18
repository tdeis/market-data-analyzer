import pandas as pd
import numpy as np


# =========================
# PRICES VALIDATION
# =========================
def validate_prices(df: pd.DataFrame):

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Prices must be a DataFrame")

    if df.empty:
        raise ValueError("Prices DataFrame is empty")

    if df.isnull().any().any():
        raise ValueError("Prices contain NaN values")

    return df.sort_index()


# =========================
# RETURNS STANDARDIZATION
# =========================
def compute_returns(prices: pd.DataFrame):

    validate_prices(prices)

    returns = prices.pct_change().dropna()

    return returns


# =========================
# WEIGHTS VALIDATION
# =========================
def validate_weights(w):

    w = np.asarray(w)

    if w.ndim != 1:
        raise ValueError("Weights must be 1D array")

    if not np.isclose(w.sum(), 1):
        raise ValueError("Weights must sum to 1")

    if np.any(w < -1e-8):
        raise ValueError("Weights contain invalid negative values")

    return w


# =========================
# ALIGN RETURNS + WEIGHTS
# =========================
def align_returns(returns: pd.DataFrame, weights: np.ndarray):

    validate_weights(weights)

    if returns.shape[1] != len(weights):
        raise ValueError("Mismatch: assets vs weights")

    return returns, weights
