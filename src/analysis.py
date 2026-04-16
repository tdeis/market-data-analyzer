def calculate_volatility(returns):
    return returns.std()


def calculate_correlation(returns):
    return returns.corr()
