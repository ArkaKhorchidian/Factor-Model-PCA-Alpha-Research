"""
Performance tearsheet generation.

Produces a multi-panel figure summarizing:
  1. Cumulative gross and net returns.
  2. Rolling Sharpe ratio (252-day trailing).
  3. Drawdown time series.
  4. Monthly return heatmap.
  5. IC time series with rolling mean.
  6. Turnover histogram.

The sample tearsheet saved to results/tearsheet_sample.png is generated
by calling generate_tearsheet() with sample data and saving the figure.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Individual panel helpers
# ---------------------------------------------------------------------------


def _plot_cumulative(ax: plt.Axes, gross: pd.Series, net: pd.Series) -> None:
    cum_gross = (1 + gross).cumprod() * 100
    cum_net = (1 + net).cumprod() * 100
    ax.plot(cum_gross.index, cum_gross.values, label="Gross", lw=1.5, color="#1f77b4")
    ax.plot(cum_net.index, cum_net.values, label="Net", lw=1.5, color="#ff7f0e", ls="--")
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Cumulative Returns (starting $100)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


def _plot_rolling_sharpe(ax: plt.Axes, net: pd.Series, window: int = 252) -> None:
    excess = net  # assume RF ≈ 0 for display
    rolling_sr = (
        excess.rolling(window).mean()
        / excess.rolling(window).std(ddof=1)
        * np.sqrt(252)
    )
    ax.plot(rolling_sr.index, rolling_sr.values, lw=1.2, color="#2ca02c")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(1, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Rolling Sharpe (252d)")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


def _plot_drawdown(ax: plt.Axes, net: pd.Series) -> None:
    wealth = (1 + net.dropna()).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax() * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.6)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


def _plot_monthly_heatmap(ax: plt.Axes, net: pd.Series) -> None:
    monthly = (1 + net).resample("ME").prod() - 1
    monthly_df = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    ).pivot(index="year", columns="month", values="ret")
    monthly_df.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    sns.heatmap(
        monthly_df * 100,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Return (%)"},
        annot_kws={"size": 6},
    )
    ax.set_title("Monthly Returns (%)")
    ax.set_xlabel("")


def _plot_ic_series(ax: plt.Axes, ic: pd.Series | None) -> None:
    if ic is None or len(ic.dropna()) == 0:
        ax.text(0.5, 0.5, "IC data not provided", ha="center", va="center",
                transform=ax.transAxes)
        return
    ax.bar(ic.index, ic.values, color=["#1f77b4" if v >= 0 else "#d62728" for v in ic.values],
           width=5, alpha=0.7, label="Daily IC")
    rolling_ic = ic.rolling(63, min_periods=21).mean()
    ax.plot(rolling_ic.index, rolling_ic.values, color="black", lw=1.5, label="63d MA")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("IC")
    ax.set_title("Information Coefficient")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


def _plot_turnover(ax: plt.Axes, to: pd.Series | None) -> None:
    if to is None or len(to.dropna()) == 0:
        ax.text(0.5, 0.5, "Turnover data not provided", ha="center", va="center",
                transform=ax.transAxes)
        return
    ax.hist(to.dropna().values * 100, bins=30, color="#9467bd", alpha=0.7, edgecolor="white")
    ax.axvline(to.mean() * 100, color="red", lw=1.5, ls="--", label=f"Mean {to.mean()*100:.1f}%")
    ax.set_xlabel("Turnover (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Portfolio Turnover Distribution")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main tearsheet
# ---------------------------------------------------------------------------


def generate_tearsheet(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    ic_series: pd.Series | None = None,
    turnover: pd.Series | None = None,
    gross_metrics: dict | None = None,
    net_metrics: dict | None = None,
    strategy_name: str = "Strategy",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 20),
) -> plt.Figure:
    """
    Generate a full performance tearsheet.

    Parameters
    ----------
    gross_returns : pd.Series
        Gross daily returns.
    net_returns : pd.Series
        Net daily returns (after transaction costs).
    ic_series : pd.Series, optional
        Daily IC values.
    turnover : pd.Series, optional
        Daily turnover series.
    gross_metrics : dict, optional
        Pre-computed gross metrics (from metrics.performance_metrics).
    net_metrics : dict, optional
        Pre-computed net metrics.
    strategy_name : str
    save_path : str or Path, optional
        If provided, save figure to this path.
    figsize : tuple[int, int]

    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"{strategy_name} — Performance Tearsheet",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Grid: 3 columns, variable rows
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    ax_cum = fig.add_subplot(gs[0, :])
    ax_sharpe = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_heatmap = fig.add_subplot(gs[2, :])
    ax_ic = fig.add_subplot(gs[3, 0])
    ax_to = fig.add_subplot(gs[3, 1])

    _plot_cumulative(ax_cum, gross_returns, net_returns)
    _plot_rolling_sharpe(ax_sharpe, net_returns)
    _plot_drawdown(ax_dd, net_returns)
    _plot_monthly_heatmap(ax_heatmap, net_returns)
    _plot_ic_series(ax_ic, ic_series)
    _plot_turnover(ax_to, turnover)

    # Metrics summary text box
    if gross_metrics and net_metrics:
        txt_lines = [
            f"{'Metric':<28} {'Gross':>10} {'Net':>10}",
            "─" * 50,
        ]
        keys = [
            ("annualized_return", "{:.2%}"),
            ("annualized_volatility", "{:.2%}"),
            ("sharpe_ratio", "{:.3f}"),
            ("sortino_ratio", "{:.3f}"),
            ("max_drawdown", "{:.2%}"),
            ("calmar_ratio", "{:.3f}"),
            ("win_rate", "{:.2%}"),
            ("avg_ic", "{:.4f}"),
            ("ic_ir", "{:.3f}"),
            ("avg_turnover", "{:.2%}"),
        ]
        for k, fmt in keys:
            g_val = gross_metrics.get(k, np.nan)
            n_val = net_metrics.get(k, np.nan)
            try:
                g_str = fmt.format(g_val) if not np.isnan(g_val) else "N/A"
                n_str = fmt.format(n_val) if not np.isnan(n_val) else "N/A"
            except (ValueError, TypeError):
                g_str = str(g_val)
                n_str = str(n_val)
            txt_lines.append(f"{k:<28} {g_str:>10} {n_str:>10}")
        txt = "\n".join(txt_lines)
        fig.text(
            0.01, 0.01, txt,
            fontfamily="monospace",
            fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_sample_tearsheet(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    **kwargs,
) -> Path:
    """Save the tearsheet to results/tearsheet_sample.png."""
    path = _RESULTS_DIR / "tearsheet_sample.png"
    generate_tearsheet(gross_returns, net_returns, save_path=path, **kwargs)
    return path
