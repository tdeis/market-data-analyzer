import numpy as np


# =========================
# VALIDATION
# =========================
def _validate_inputs(weights, vector, name="input"):

    weights = np.asarray(weights)
    vector = np.asarray(vector)

    if weights.ndim != 1:
        raise ValueError("weights must be 1D array")

    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D array")

    if len(weights) != len(vector):
        raise ValueError("weights and vector must have same length")

    if not np.isfinite(weights).all():
        raise ValueError("weights contain NaN or inf")

    return weights, vector


# =========================
# RETURN
# =========================
def portfolio_return(weights, returns):

    w, r = _validate_inputs(weights, returns, "returns")

    return float(np.dot(w, r))


# =========================
# VOLATILITY (STABLE)
# =========================
def portfolio_volatility(weights, cov_matrix):

    w = np.asarray(weights)
    cov = np.asarray(cov_matrix)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be square")

    if cov.shape[0] != len(w):
        raise ValueError("cov_matrix size must match weights")

    if not np.isfinite(cov).all():
        raise ValueError("cov_matrix contains NaN or inf")

    var = w.T @ cov @ w

    if var < 0:
        var = max(var, 0)  # numerical safety

    return float(np.sqrt(var))


# =========================
# SHARPE RATIO (ROBUST)
# =========================
def sharpe_ratio(weights, returns, cov_matrix, risk_free=0.0):

    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov_matrix)

    if vol < 1e-10:
        return 0.0

    return (ret - risk_free) / vol


# =========================
# NEGATIVE SHARPE (FOR OPTIMIZER)
# =========================
def negative_sharpe(weights, returns, cov_matrix):

    try:
        return -sharpe_ratio(weights, returns, cov_matrix)
    except Exception:
        return 1e6


# =========================
# DIVERSIFICATION METRIC
# =========================
def diversification_ratio(weights):

    w = np.asarray(weights)

    if len(w) == 0:
        raise ValueError("weights cannot be empty")

    # Herfindahl index (concentration measure)
    hhi = np.sum(w**2)

    return float(1 / hhi if hhi > 0 else 0)


# =========================
# CONCENTRATION PENALTY
# =========================
def concentration_penalty(weights):

    w = np.asarray(weights)
    return float(np.sum(w**2))


# =========================
# ENHANCED OBJECTIVE (REGULARIZED)
# =========================
def negative_sharpe_defensive(weights, returns, cov_matrix):

    try:
        ret = portfolio_return(weights, returns)
        vol = portfolio_volatility(weights, cov_matrix)

        if vol < 1e-10:
            return 1e6

        sharpe = ret / vol
        penalty = concentration_penalty(weights)

        # encourages diversification
        return -(sharpe - 0.15 * penalty)

    except Exception:
        return 1e6


# =========================
# PORTFOLIO METRICS SUMMARY
# =========================
def portfolio_metrics(weights, returns, cov_matrix):

    return {
        "return": portfolio_return(weights, returns),
        "volatility": portfolio_volatility(weights, cov_matrix),
        "sharpe": sharpe_ratio(weights, returns, cov_matrix),
        "diversification": diversification_ratio(weights),
        "concentration": concentration_penalty(weights),
    }
