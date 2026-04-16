import pandas as pd


def calculate_returns(prices: pd.DataFrame):
    return prices.pct_change().dropna()


def normalize(prices: pd.DataFrame):
    return prices / prices.iloc[0]
