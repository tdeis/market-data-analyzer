import numpy as np
import pandas as pd


def compute_returns(prices):

    return prices.pct_change().dropna()
