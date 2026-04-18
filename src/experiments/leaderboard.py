import pandas as pd
from src.core.experiment import ExperimentResult


def build_leaderboard(results):
    """
    Builds a leaderboard from ExperimentResult objects.
    Now includes defensive handling for legacy / broken types.
    """

    rows = []

    for r in results:

        # -------------------------
        # DEFENSIVE TYPE HANDLING
        # -------------------------
        if isinstance(r, list):
            raise TypeError(
                "Leaderboard received a list instead of ExperimentResult. "
                "Check run_all() return type."
            )

        if not isinstance(r, ExperimentResult):
            raise TypeError(
                f"Invalid result type: {type(r)}. Expected ExperimentResult."
            )

        # -------------------------
        # SAFE EXTRACTION
        # -------------------------
        metrics = r.metrics or {}

        rows.append(
            {
                "strategy": r.strategy,
                "sharpe": metrics.get("sharpe", 0.0),
                "volatility": metrics.get("volatility", 0.0),
                "drawdown": metrics.get("max_drawdown", 0.0),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(by="sharpe", ascending=False).reset_index(drop=True)
