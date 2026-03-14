"""
Performance metrics for strategy evaluation.

All functions accept a pd.Series of period returns (daily by default).
Annualization assumes 252 trading days per year unless specified.

References
----------
Grinold & Kahn (1999), Ch. 6 — Sharpe, IC, IR.
Sortino, F. & van der Meer, R. (1991) — Sortino ratio.
Calmar, T. (1991) — Calmar ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return: (1 + r̄)^periods_per_year - 1."""
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    total = (1 + returns.dropna()).prod()
    return float(total ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio: (E[R - Rf]) / σ(R - Rf) × √periods_per_year.

    Parameters
    ----------
    returns : pd.Series
        Period returns (not excess).
    risk_free : float or pd.Series
        Risk-free rate per period.  If 0.0, gross Sharpe is returned.
    periods_per_year : int

    Returns
    -------
    float
    """
    excess = returns - risk_free if isinstance(risk_free, pd.Series) else returns - risk_free
    excess = excess.dropna()
    if len(excess) < 2:
        return np.nan
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma < 1e-10:
        return 0.0 if abs(mu) < 1e-10 else float(np.sign(mu) * np.inf)
    return float(mu / sigma * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sortino ratio: E[R - Rf] / downside_std × √periods_per_year.

    Downside std uses only negative excess returns.
    """
    excess = (returns - risk_free).dropna()
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    ann_ret = excess.mean() * periods_per_year
    return float(ann_ret / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (as a negative fraction).

    Returns
    -------
    float
        MDD in decimal (e.g., -0.25 for a 25% drawdown).
    """
    wealth = (1 + returns.dropna()).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio: annualized_return / |max_drawdown|.

    Higher is better.  A value > 1.0 is considered strong.
    """
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return np.nan
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return."""
    r = returns.dropna()
    return float((r > 0).mean())


def average_win_loss(returns: pd.Series) -> float:
    """Ratio of average winning return to average absolute losing return."""
    r = returns.dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    if len(losses) == 0:
        return np.nan
    return float(wins.mean() / abs(losses.mean()))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.

    Returns the loss at the (1 - confidence) quantile (negative number).
    """
    return float(returns.dropna().quantile(1 - confidence))


def turnover(weights_history: pd.DataFrame) -> pd.Series:
    """
    Compute per-period portfolio turnover as fraction of portfolio turned over.

    Turnover at t = 0.5 × Σ|w_t - w_{t-1}| (each leg counts once).

    Parameters
    ----------
    weights_history : pd.DataFrame
        Portfolio weights, shape (T, N).

    Returns
    -------
    pd.Series
        Per-period turnover, shape (T,).
    """
    delta = weights_history.diff().abs().sum(axis=1) / 2
    return delta.rename("turnover")


# ---------------------------------------------------------------------------
# IC / signal quality
# ---------------------------------------------------------------------------


def ic_ir(ic_series: pd.Series) -> float:
    """
    Information ratio of the IC series: mean(IC) / std(IC).

    Reference: Grinold & Kahn (1999), Eq. (6.4).
    """
    ic = ic_series.dropna()
    sigma = ic.std(ddof=1)
    if sigma < 1e-10:
        return np.nan
    return float(ic.mean() / sigma)


# ---------------------------------------------------------------------------
# Full performance report
# ---------------------------------------------------------------------------


def performance_metrics(
    returns: pd.Series,
    risk_free: pd.Series | float = 0.0,
    ic_series: pd.Series | None = None,
    weights_history: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Compute the full set of performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Period returns of the strategy.
    risk_free : pd.Series or float
        Risk-free rate per period.
    ic_series : pd.Series, optional
        Pre-computed IC time series for IC/IR reporting.
    weights_history : pd.DataFrame, optional
        Portfolio weights for turnover calculation.
    periods_per_year : int

    Returns
    -------
    dict[str, float]
        Keys: annualized_return, annualized_volatility, sharpe_ratio,
              sortino_ratio, max_drawdown, calmar_ratio, win_rate,
              avg_ic, ic_ir, avg_turnover.
    """
    metrics: dict[str, float] = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino_ratio": sortino_ratio(
            returns,
            risk_free if isinstance(risk_free, float) else float(risk_free.mean()),
            periods_per_year,
        ),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "value_at_risk_95": value_at_risk(returns, 0.95),
    }

    if ic_series is not None:
        ic_clean = ic_series.dropna()
        metrics["avg_ic"] = float(ic_clean.mean()) if len(ic_clean) > 0 else np.nan
        metrics["ic_ir"] = ic_ir(ic_series)
    else:
        metrics["avg_ic"] = np.nan
        metrics["ic_ir"] = np.nan

    if weights_history is not None:
        to = turnover(weights_history)
        metrics["avg_turnover"] = float(to.mean())
    else:
        metrics["avg_turnover"] = np.nan

    return metrics


def format_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    """Pretty-print metrics as a single-column DataFrame."""
    formatters = {
        "annualized_return": "{:.2%}",
        "annualized_volatility": "{:.2%}",
        "sharpe_ratio": "{:.3f}",
        "sortino_ratio": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "calmar_ratio": "{:.3f}",
        "win_rate": "{:.2%}",
        "value_at_risk_95": "{:.2%}",
        "avg_ic": "{:.4f}",
        "ic_ir": "{:.3f}",
        "avg_turnover": "{:.2%}",
    }
    rows = []
    for key, val in metrics.items():
        fmt = formatters.get(key, "{:.4f}")
        try:
            display = fmt.format(val)
        except (ValueError, TypeError):
            display = str(val)
        rows.append({"Metric": key, "Value": display})
    return pd.DataFrame(rows).set_index("Metric")
