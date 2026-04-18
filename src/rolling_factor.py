import pandas as pd
import statsmodels.api as sm


def rolling_alpha_beta(strategy_returns, market_returns, window=60):

    combined = pd.concat([strategy_returns, market_returns], axis=1).dropna()

    if len(combined) < window:
        raise ValueError("Not enough data for rolling regression")

    combined.columns = ["strategy", "market"]

    alphas = []
    betas = []
    dates = []

    for i in range(window, len(combined)):

        window_df = combined.iloc[i - window : i]

        X = sm.add_constant(window_df["market"])
        y = window_df["strategy"]

        model = sm.OLS(y, X).fit()

        alphas.append(model.params["const"])
        betas.append(model.params["market"])
        dates.append(combined.index[i])

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=dates)
