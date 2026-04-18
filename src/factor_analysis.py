import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================
# VALIDATION HELPERS
# =========================
def _validate_inputs(strategy_returns, market_returns):
    if not isinstance(strategy_returns, pd.Series):
        raise TypeError("strategy_returns must be a pandas Series")

    if not isinstance(market_returns, pd.Series):
        raise TypeError("market_returns must be a pandas Series")

    if strategy_returns.empty or market_returns.empty:
        raise ValueError("Input series cannot be empty")

    if strategy_returns.isnull().all() or market_returns.isnull().all():
        raise ValueError("Input series contain only NaNs")


def _align_series(strategy_returns, market_returns):
    df = pd.concat([strategy_returns, market_returns], axis=1).dropna()

    if df.empty:
        raise ValueError("No overlapping data after alignment")

    df.columns = ["strategy", "market"]
    return df["strategy"], df["market"]


# =========================
# STATIC FACTOR MODEL
# =========================
def compute_beta_alpha(strategy_returns, market_returns):

    _validate_inputs(strategy_returns, market_returns)

    strat, mkt = _align_series(strategy_returns, market_returns)

    if len(strat) < 20:
        raise ValueError("Not enough data for regression (need >= 20 points)")

    X = sm.add_constant(mkt)
    y = strat

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise RuntimeError(f"OLS regression failed: {e}")

    # Safe extraction
    alpha = float(model.params.get("const", np.nan))
    beta = float(model.params.get("market", np.nan))
    r2 = float(model.rsquared)

    return {"alpha": alpha, "beta": beta, "r_squared": r2}


# =========================
# ROLLING FACTOR EXPOSURE
# =========================
def rolling_alpha_beta(strategy_returns, market_returns, window=60):

    _validate_inputs(strategy_returns, market_returns)

    strat, mkt = _align_series(strategy_returns, market_returns)

    if len(strat) < window:
        raise ValueError("Not enough data for rolling regression window")

    alphas = np.full(len(strat), np.nan)
    betas = np.full(len(strat), np.nan)

    index = strat.index

    for i in range(window, len(strat)):

        y = strat.iloc[i - window : i]
        x = mkt.iloc[i - window : i]

        X = sm.add_constant(x)

        try:
            model = sm.OLS(y, X).fit()

            alphas[i] = model.params.get("const", np.nan)
            betas[i] = model.params.get("market", np.nan)

        except Exception:
            # Fail gracefully instead of crashing pipeline
            continue

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=index)


# =========================
# FACTOR SUMMARY WRAPPER
# =========================
def factor_summary(strategy_returns, market_returns, window=60):

    static = compute_beta_alpha(strategy_returns, market_returns)
    rolling = rolling_alpha_beta(strategy_returns, market_returns, window)

    return {"static": static, "rolling": rolling}
