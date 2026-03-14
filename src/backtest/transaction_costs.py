"""
Transaction cost models: slippage and commissions.

Two models are provided:
  1. Flat cost model: fixed basis points per one-way trade (fast, conservative).
  2. Proportional cost model: cost scales with trade size as fraction of ADV.

For large-cap equities with monthly rebalancing, 5 bps one-way is conservative.
For small-caps or high-frequency strategies, use 10–20 bps.

Always report gross AND net Sharpe.  If the spread collapses on a net basis,
that is a legitimate research finding, not a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_flat_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    one_way_cost_bps: float = 5.0,
) -> pd.Series:
    """
    Apply a flat transaction cost model.

    Net return = gross return - 2 × one_way_cost × turnover_fraction.

    The factor of 2 accounts for both sides of a round trip (buy + sell).

    Parameters
    ----------
    gross_returns : pd.Series
        Gross strategy returns, shape (T,).
    turnover : pd.Series
        Per-period portfolio turnover (fraction of portfolio traded), shape (T,).
        Computed as 0.5 × Σ|Δw_i|.
    one_way_cost_bps : float
        One-way transaction cost in basis points (default 5 bps).

    Returns
    -------
    pd.Series
        Net returns after transaction costs.
    """
    one_way_cost = one_way_cost_bps / 10_000  # bps → decimal
    cost = 2 * one_way_cost * turnover
    aligned_cost = cost.reindex(gross_returns.index).fillna(0.0)
    return (gross_returns - aligned_cost).rename("net_return")


def apply_proportional_costs(
    gross_returns: pd.Series,
    trade_sizes: pd.DataFrame,
    adv: pd.DataFrame,
    market_impact_coeff: float = 0.5,
    fixed_cost_bps: float = 2.0,
) -> pd.Series:
    """
    Apply a proportional market impact cost model.

    Cost per trade = fixed_cost_bps + market_impact_coeff × (trade_size / ADV).

    This is a simplified version of the Almgren-Chriss permanent impact term.

    Parameters
    ----------
    gross_returns : pd.Series
        Gross strategy returns.
    trade_sizes : pd.DataFrame
        Notional trade sizes per stock per period, shape (T, N).
    adv : pd.DataFrame
        Average daily volume per stock, shape (T, N).
    market_impact_coeff : float
        Linear market impact coefficient.
    fixed_cost_bps : float
        Fixed per-trade cost in basis points.

    Returns
    -------
    pd.Series
        Net returns after market impact.
    """
    pov = trade_sizes.abs() / adv.replace(0, np.nan)  # participation of volume
    impact_bps = fixed_cost_bps + market_impact_coeff * pov * 10_000
    total_cost = (impact_bps / 10_000 * trade_sizes.abs()).sum(axis=1)
    # Normalize cost as fraction of total portfolio value
    portfolio_value = trade_sizes.abs().sum(axis=1)
    cost_return = total_cost / portfolio_value.replace(0, np.nan)
    aligned_cost = cost_return.reindex(gross_returns.index).fillna(0.0)
    return (gross_returns - aligned_cost).rename("net_return_proportional")


def cost_drag(
    gross_sharpe: float,
    net_sharpe: float,
) -> dict[str, float]:
    """
    Summarize the cost drag between gross and net performance.

    Parameters
    ----------
    gross_sharpe : float
    net_sharpe : float

    Returns
    -------
    dict
        Keys: gross_sharpe, net_sharpe, sharpe_drag, drag_pct.
    """
    drag = gross_sharpe - net_sharpe
    drag_pct = drag / gross_sharpe if gross_sharpe != 0 else np.nan
    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "sharpe_drag": drag,
        "drag_pct": drag_pct,
    }
