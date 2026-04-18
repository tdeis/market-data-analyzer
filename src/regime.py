import numpy as np
import pandas as pd


def detect_regime(prices, benchmark="SPY", window=200):

    if benchmark not in prices.columns:
        raise ValueError(f"{benchmark} not found in prices")

    if len(prices) < window:
        raise ValueError("Not enough data for regime detection")

    series = prices[benchmark]
    ma = series.rolling(window).mean()

    regime = np.where(series > ma, "bull", "bear")

    return pd.Series(regime, index=prices.index)
