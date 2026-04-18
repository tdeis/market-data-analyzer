from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.core.schema import compute_returns


# =========================
# SAFE RETURNS HANDLER
# =========================
def _ensure_series(x):

    if isinstance(x, pd.DataFrame):
        return x.mean(axis=1)

    if isinstance(x, pd.Series):
        return x

    return pd.Series(x)


# =========================
# MAIN FIGURE PIPELINE
# =========================
def generate_figures(prices, returns, save_dir="outputs/figures"):

    returns = returns.squeeze()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # NORMALIZE INPUT
    # =========================
    returns = _ensure_series(returns)

    # =========================
    # EQUITY CURVE
    # =========================
    equity = (1 + returns).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "equity_curve.png")
    plt.close()

    # =========================
    # DRAWDOWN
    # =========================
    peak = equity.cummax()
    drawdown = equity / peak - 1

    plt.figure(figsize=(10, 5))
    plt.plot(drawdown, color="red")
    plt.title("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "drawdown.png")
    plt.close()

    # =========================
    # RETURNS DISTRIBUTION
    # =========================
    plt.figure(figsize=(10, 5))
    returns.hist(bins=50)
    plt.title("Return Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "returns_distribution.png")
    plt.close()

    print(f"📊 Figures saved to {save_dir}")
