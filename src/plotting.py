import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path


# =========================
# GLOBAL STYLE
# =========================
plt.style.use("seaborn-v0_8")


# =========================
# VALIDATION
# =========================
def _validate_series(series, name="series"):
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")

    if series.empty:
        raise ValueError(f"{name} is empty")

    if series.isnull().all():
        raise ValueError(f"{name} contains only NaNs")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================
# ALIGNMENT
# =========================
def _align(*series_list):
    df = pd.concat(series_list, axis=1).dropna()

    if df.empty:
        raise ValueError("No overlapping data after alignment")

    return [df[col] for col in df.columns]


# =========================
# EQUITY CURVE
# =========================
def plot_equity_curve(strategy_returns, benchmark_returns=None, save_path=None):

    _validate_series(strategy_returns, "strategy_returns")

    series_list = [strategy_returns]

    if benchmark_returns is not None:
        _validate_series(benchmark_returns, "benchmark_returns")
        strategy_returns, benchmark_returns = _align(
            strategy_returns, benchmark_returns
        )
        series_list = [strategy_returns, benchmark_returns]

    equity = [(1 + s).cumprod() for s in series_list]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity[0], label="Strategy", linewidth=2)

    if len(equity) > 1:
        ax.plot(equity[1], label="Benchmark", linestyle="--")

    ax.set_title("Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# =========================
# DRAWDOWN
# =========================
def plot_drawdown(strategy_returns, save_path=None):

    _validate_series(strategy_returns, "strategy_returns")

    equity = (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(drawdown, color="red")
    ax.set_title("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# =========================
# ROLLING METRIC (GENERIC)
# =========================
def plot_rolling(series, title="Rolling Metric", window=60, save_path=None):

    _validate_series(series, "series")

    rolling = series.rolling(window).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(rolling)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# =========================
# WEIGHTS OVER TIME
# =========================
def plot_weights(weights_df, save_path=None):

    if not isinstance(weights_df, pd.DataFrame):
        raise TypeError("weights_df must be DataFrame")

    if weights_df.empty:
        raise ValueError("weights_df is empty")

    fig, ax = plt.subplots(figsize=(12, 6))

    weights_df.plot(ax=ax, linewidth=1)

    ax.set_title("Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# =========================
# DASHBOARD
# =========================
def plot_dashboard(strategy_returns, benchmark_returns=None, save_path=None):

    _validate_series(strategy_returns, "strategy_returns")

    if benchmark_returns is not None:
        _validate_series(benchmark_returns, "benchmark_returns")
        strategy_returns, benchmark_returns = _align(
            strategy_returns, benchmark_returns
        )

    equity = (1 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    rolling_sharpe = (
        strategy_returns.rolling(60).mean() / strategy_returns.rolling(60).std()
    ) * np.sqrt(252)

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Equity
    ax[0].plot(equity, label="Strategy")
    if benchmark_returns is not None:
        bench_eq = (1 + benchmark_returns).cumprod()
        ax[0].plot(bench_eq, linestyle="--", label="Benchmark")

    ax[0].set_title("Equity Curve")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Drawdown
    ax[1].plot(drawdown, color="red")
    ax[1].set_title("Drawdown")
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1].grid(True, alpha=0.3)

    # Sharpe
    ax[2].plot(rolling_sharpe, color="purple")
    ax[2].axhline(0, linestyle="--")
    ax[2].set_title("Rolling Sharpe (60D)")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
