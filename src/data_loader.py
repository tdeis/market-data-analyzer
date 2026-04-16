import yfinance as yf


def get_prices(tickers, start="2020-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)
    return data["Close"]
