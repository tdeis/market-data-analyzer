import pandas as pd
import yfinance as yf


def get_prices(tickers, start="2015-01-01", end=None):

    if not isinstance(tickers, list) or not tickers:
        raise ValueError("tickers must be a non-empty list")

    try:
        data = yf.download(tickers, start=start, end=end, progress=False)
    except Exception as e:
        raise RuntimeError(f"Data download failed: {e}")

    if data.empty:
        raise ValueError("No data returned from yfinance")

    # -------------------------
    # HANDLE COLUMN STRUCTURE
    # -------------------------
    if isinstance(data.columns, pd.MultiIndex):
        # multi-ticker case
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            print("⚠️ 'Adj Close' missing, using 'Close' instead")
            prices = data["Close"]
        else:
            raise KeyError("No usable price column found")

    else:
        # single ticker case
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            print("⚠️ 'Adj Close' missing, using 'Close' instead")
            prices = data["Close"]
        else:
            raise KeyError("No usable price column found")

    # -------------------------
    # CLEAN DATA
    # -------------------------
    prices = prices.ffill().dropna()

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    return prices
