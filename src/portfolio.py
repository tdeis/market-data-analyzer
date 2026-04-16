import numpy as np


def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_return = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)

    return (p_return - risk_free_rate) / p_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
