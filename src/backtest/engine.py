"""
Event-driven backtest loop.

The engine iterates through time, applies portfolio weights to realised returns,
and accumulates gross and net P&L.  All inputs are pre-computed; no fitting
or signal generation happens inside the engine (avoids look-ahead bias).

Design principles:
  - Pure function — no hidden state.
  - Dollar-neutral at each rebalance.
  - Reports gross and net separately (transaction cost impact is explicit).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.metrics import performance_metrics, turnover as compute_turnover
from src.backtest.transaction_costs import apply_flat_costs

logger = logging.getLogger(__name__)


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
    one_way_cost_bps: float = 5.0,
    rebalance_freq: str = "ME",
    name: str = "Strategy",
) -> dict:
    """
    Core backtest loop.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights, shape (T, N).  Must be aligned with *returns*.
        Weights are applied to the next period's return (no look-ahead).
    returns : pd.DataFrame
        Realised daily returns, shape (T, N).
    risk_free : pd.Series or float
        Risk-free rate per period (same frequency as returns).
    one_way_cost_bps : float
        One-way transaction cost in basis points.
    rebalance_freq : str
        Frequency at which weights change (for documentation/diagnostics).
    name : str
        Strategy name (used in output keys).

    Returns
    -------
    dict with keys:
        gross_returns : pd.Series
        net_returns   : pd.Series
        turnover      : pd.Series
        weights       : pd.DataFrame (aligned)
        gross_metrics : dict
        net_metrics   : dict
    """
    # Align weights and returns on common dates and tickers
    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)

    if len(common_dates) < 10:
        raise ValueError(
            f"Insufficient overlapping dates: {len(common_dates)}. "
            "Check alignment of weights and returns DataFrames."
        )

    w = weights.loc[common_dates, common_tickers].fillna(0.0)
    r = returns.loc[common_dates, common_tickers].fillna(0.0)

    # Shift weights by 1: signal observed on date t, positions established at
    # close of t, return realised from close t to close t+1.
    w_shifted = w.shift(1).fillna(0.0)

    # Gross portfolio return: Σ w_i × r_i (daily)
    gross_ret = (w_shifted * r).sum(axis=1)
    gross_ret.name = "gross_return"

    # Turnover: |Δw| at each period
    to = compute_turnover(w_shifted)

    # Net return: deduct transaction costs
    net_ret = apply_flat_costs(gross_ret, to, one_way_cost_bps)

    # Align risk-free rate
    if isinstance(risk_free, pd.Series):
        rf_aligned = risk_free.reindex(common_dates).fillna(0.0)
    else:
        rf_aligned = risk_free

    # Compute metrics
    gross_m = performance_metrics(gross_ret, rf_aligned)
    net_m = performance_metrics(net_ret, rf_aligned)

    logger.info(
        "%s — Gross Sharpe: %.3f | Net Sharpe: %.3f | Avg Turnover: %.1f%%",
        name,
        gross_m["sharpe_ratio"],
        net_m["sharpe_ratio"],
        to.mean() * 100,
    )

    return {
        "gross_returns": gross_ret,
        "net_returns": net_ret,
        "turnover": to,
        "weights": w,
        "gross_metrics": gross_m,
        "net_metrics": net_m,
        "name": name,
    }


def quintile_backtest(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    n_quintiles: int = 5,
    rebalance_freq: str = "ME",
    one_way_cost_bps: float = 5.0,
    risk_free: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """
    Run a quintile-sorted backtest for each quintile bucket independently.

    Useful for diagnosing monotonicity: does alpha increase from Q1 to Q5?

    Parameters
    ----------
    signal : pd.DataFrame
        Alpha signal, shape (T, N).
    returns : pd.DataFrame
        Realised returns, shape (T, N).
    n_quintiles : int
    rebalance_freq : str
    one_way_cost_bps : float
    risk_free : pd.Series or float

    Returns
    -------
    pd.DataFrame
        Cumulative gross returns for each quintile, shape (T, n_quintiles).
    """
    from src.backtest.portfolio import build_portfolio_weights

    quintile_returns = {}
    for q in range(1, n_quintiles + 1):
        w = build_portfolio_weights(
            signal,
            rebalance_freq=rebalance_freq,
            n_quintiles=n_quintiles,
            weight_scheme="equal",
        )
        # Reconstruct single-quintile weights (long only)
        # Re-rank within each date
        rets_list = []
        reb_dates = signal.resample(rebalance_freq).last().index
        current_q_weights: pd.Series | None = None

        for date in returns.index:
            if date in reb_dates or current_q_weights is None:
                sig_row = signal.loc[date].dropna()
                if len(sig_row) >= n_quintiles:
                    ranked = sig_row.rank(pct=True)
                    lo = (q - 1) / n_quintiles
                    hi = q / n_quintiles
                    in_q = ranked[(ranked > lo) & (ranked <= hi)].index
                    if len(in_q) > 0:
                        current_q_weights = pd.Series(
                            1.0 / len(in_q), index=in_q
                        ).reindex(returns.columns).fillna(0.0)
                    else:
                        current_q_weights = pd.Series(0.0, index=returns.columns)
                else:
                    current_q_weights = pd.Series(0.0, index=returns.columns)

            if current_q_weights is not None:
                ret = (current_q_weights * returns.loc[date]).sum()
                rets_list.append(ret)
            else:
                rets_list.append(0.0)

        quintile_returns[f"Q{q}"] = pd.Series(rets_list, index=returns.index)

    df = pd.DataFrame(quintile_returns)
    return (1 + df).cumprod()


def cumulative_returns(returns: pd.Series, start_value: float = 100.0) -> pd.Series:
    """
    Convert period returns to a cumulative wealth index starting at *start_value*.
    """
    return ((1 + returns.dropna()).cumprod() * start_value).rename("cumulative")
