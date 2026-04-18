from scipy.optimize import minimize
import numpy as np


def optimize_portfolio(mu, cov, bounds, constraints, init_guess, objective):

    if len(mu) == 0:
        raise ValueError("Empty mean returns")

    try:
        res = minimize(
            objective,
            init_guess,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    except Exception as e:
        raise RuntimeError(f"Optimizer crashed: {e}")

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x

    if not np.isfinite(w).all():
        raise ValueError("Invalid weights returned")

    return w
