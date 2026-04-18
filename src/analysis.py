import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================
# VALIDATION HELPERS
# =========================
def _validate_series(x, name):
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")

    if x.empty:
        raise ValueError(f"{name} is empty")

    if x.isnull().all():
        raise ValueError(f"{name} contains only NaNs")


def _align(strategy, market):
    df = pd.concat([strategy, market], axis=1).dropna()

    if df.empty:
        raise ValueError("No overlapping data after alignment")

    df.columns = ["strategy", "market"]
    return df["strategy"], df["market"]


# =========================
# STATIC FACTOR REGRESSION
# =========================
def compute_beta_alpha(strategy_returns, market_returns):

    _validate_series(strategy_returns, "strategy_returns")
    _validate_series(market_returns, "market_returns")

    strat, mkt = _align(strategy_returns, market_returns)

    if len(strat) < 10:
        raise ValueError("Not enough data for regression")

    X = sm.add_constant(mkt)
    y = strat

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise RuntimeError(f"OLS regression failed: {e}")

    return {
        "alpha": float(model.params["const"]),
        "beta": float(model.params["market"]),
        "r_squared": float(model.rsquared),
    }


# =========================
# ROLLING FACTOR EXPOSURE
# =========================
def rolling_alpha_beta(strategy_returns, market_returns, window=60):

    _validate_series(strategy_returns, "strategy_returns")
    _validate_series(market_returns, "market_returns")

    strat, mkt = _align(strategy_returns, market_returns)

    if len(strat) < window:
        raise ValueError("Not enough data for rolling window")

    alphas = np.full(len(strat), np.nan)
    betas = np.full(len(strat), np.nan)
    index = strat.index

    for i in range(window, len(strat)):

        y = strat.iloc[i - window : i]
        x = mkt.iloc[i - window : i]

        X = sm.add_constant(x)

        try:
            model = sm.OLS(y, X).fit()
            alphas[i] = model.params["const"]
            betas[i] = model.params["market"]
        except Exception:
            # fail gracefully for unstable windows
            alphas[i] = np.nan
            betas[i] = np.nan

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=index)


# =========================
# STABILITY METRIC
# =========================
def stability_score(rolling_factors: pd.DataFrame):

    if not isinstance(rolling_factors, pd.DataFrame):
        raise TypeError("rolling_factors must be DataFrame")

    if rolling_factors.empty:
        raise ValueError("rolling_factors is empty")

    required = {"alpha", "beta"}
    if not required.issubset(rolling_factors.columns):
        raise ValueError(f"Missing columns: {required - set(rolling_factors.columns)}")

    def cv(x):
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        mean = np.mean(x)
        std = np.std(x)
        return std / abs(mean) if abs(mean) > 1e-10 else np.nan

    alpha_cv = cv(rolling_factors["alpha"])
    beta_cv = cv(rolling_factors["beta"])

    alpha_stability = 1 / (1 + alpha_cv) if alpha_cv == alpha_cv else 0
    beta_stability = 1 / (1 + beta_cv) if beta_cv == beta_cv else 0

    return {
        "alpha_stability": float(alpha_stability),
        "beta_stability": float(beta_stability),
        "overall_stability": float((alpha_stability + beta_stability) / 2),
    }


# =========================
# SUMMARY WRAPPER
# =========================
def full_factor_summary(strategy_returns, market_returns, window=60):

    static = compute_beta_alpha(strategy_returns, market_returns)
    rolling = rolling_alpha_beta(strategy_returns, market_returns, window)
    stability = stability_score(rolling)

    return {"static": static, "stability": stability, "rolling": rolling}
