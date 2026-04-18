import numpy as np


def _safe_cv(series):
    mean = series.mean()
    std = series.std()

    if abs(mean) < 1e-10:
        return np.nan

    return std / abs(mean)


def stability_score(rolling_factors):

    if rolling_factors.empty:
        raise ValueError("Empty rolling factors")

    alpha_cv = _safe_cv(rolling_factors["alpha"])
    beta_cv = _safe_cv(rolling_factors["beta"])

    alpha_stability = 1 / (1 + alpha_cv) if alpha_cv == alpha_cv else 0
    beta_stability = 1 / (1 + beta_cv) if beta_cv == beta_cv else 0

    overall = 0.5 * (alpha_stability + beta_stability)

    return {
        "alpha_stability": alpha_stability,
        "beta_stability": beta_stability,
        "overall_stability": overall,
    }
