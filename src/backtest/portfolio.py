"""
Long/short portfolio construction and rebalancing.

Implements dollar-neutral quintile portfolios:
  - Long: top quintile of alpha signal.
  - Short: bottom quintile of alpha signal.
  - Dollar-neutral: long notional ≈ short notional each period.

Also provides mean-variance optimized weights via CVXPY (optional).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quintile / rank-based construction
# ---------------------------------------------------------------------------


def quintile_weights(
    signal: pd.Series,
    n_quintiles: int = 5,
    long_quintiles: tuple[int, ...] = (5,),
    short_quintiles: tuple[int, ...] = (1,),
    weight_scheme: str = "equal",
) -> pd.Series:
    """
    Assign +/- weights based on signal quintile ranking.

    Dollar-neutral: long and short legs are each normalized to sum to ±1.

    Parameters
    ----------
    signal : pd.Series
        Cross-sectional alpha signal for one date.
    n_quintiles : int
        Number of quantile buckets (default 5).
    long_quintiles : tuple[int]
        Which buckets to go long (1 = bottom, n_quintiles = top).
    short_quintiles : tuple[int]
        Which buckets to short.
    weight_scheme : {"equal", "signal"}
        "equal" — equal weight within each leg.
        "signal" — weight proportional to signal magnitude.

    Returns
    -------
    pd.Series
        Portfolio weights.  Long ≈ +1 total, short ≈ -1 total.
    """
    valid = signal.dropna()
    if len(valid) < n_quintiles * 2:
        return pd.Series(0.0, index=signal.index)

    q = pd.qcut(valid.rank(method="first"), n_quintiles, labels=False) + 1  # 1-indexed
    weights = pd.Series(0.0, index=signal.index)

    long_mask = q[q.isin(long_quintiles)].index
    short_mask = q[q.isin(short_quintiles)].index

    if len(long_mask) == 0 or len(short_mask) == 0:
        return weights

    if weight_scheme == "equal":
        weights.loc[long_mask] = 1.0 / len(long_mask)
        weights.loc[short_mask] = -1.0 / len(short_mask)
    elif weight_scheme == "signal":
        long_signal = valid.loc[long_mask]
        short_signal = valid.loc[short_mask].abs()
        if long_signal.sum() > 0:
            weights.loc[long_mask] = long_signal / long_signal.sum()
        if short_signal.sum() > 0:
            weights.loc[short_mask] = -short_signal / short_signal.sum()
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    return weights


def build_portfolio_weights(
    signal: pd.DataFrame,
    rebalance_freq: str = "ME",
    n_quintiles: int = 5,
    weight_scheme: str = "equal",
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Build portfolio weights time series by rebalancing on a fixed schedule.

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectional alpha, shape (T, N).
    rebalance_freq : str
        Pandas resample frequency string.  "ME" = monthly, "W" = weekly.
    n_quintiles : int
    weight_scheme : str
    transaction_cost_bps : float
        Stored for reference but not applied here (see backtest.engine).

    Returns
    -------
    pd.DataFrame
        Portfolio weights, shape (T, N).  Each row sums to ~0 (dollar-neutral).
    """
    # Identify rebalance dates
    reb_dates = signal.resample(rebalance_freq).last().index

    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(0.0, index=signal.columns)

    for date in signal.index:
        if date in reb_dates or not weight_rows:
            sig_today = signal.loc[date]
            new_weights = quintile_weights(sig_today, n_quintiles, weight_scheme=weight_scheme)
            current_weights = new_weights
        weight_rows[date] = current_weights.copy()

    weights = pd.DataFrame(weight_rows).T
    weights.index = pd.DatetimeIndex(weights.index)
    weights = weights.reindex(columns=signal.columns).fillna(0.0)
    return weights


def verify_dollar_neutral(weights: pd.DataFrame, tol: float = 0.05) -> pd.Series:
    """
    Verify dollar neutrality: |Σ w_i| < tol for each period.

    Returns boolean Series; True = neutral, False = drift beyond tolerance.
    """
    net_exposure = weights.sum(axis=1).abs()
    return (net_exposure < tol).rename("is_dollar_neutral")


# ---------------------------------------------------------------------------
# Mean-variance optimization (optional, requires CVXPY)
# ---------------------------------------------------------------------------


def mv_weights(
    alpha: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    max_weight: float = 0.10,
    min_weight: float = -0.10,
    turnover_limit: float | None = None,
    prev_weights: pd.Series | None = None,
) -> pd.Series:
    """
    Mean-variance optimal weights via CVXPY.

    Objective: maximize  alpha' w - (risk_aversion / 2) × w' Σ w
    Subject to: Σ w = 0 (dollar neutral), |w_i| ≤ max_weight.

    Reference: Markowitz (1952); Grinold & Kahn (1999), Ch. 8.

    Parameters
    ----------
    alpha : pd.Series
        Expected returns (alpha signal), shape (N,).
    cov_matrix : pd.DataFrame
        Return covariance matrix, shape (N, N).
    risk_aversion : float
        Lambda in the mean-variance objective.
    max_weight : float
        Maximum long weight per stock.
    min_weight : float
        Minimum (most negative) short weight per stock.
    turnover_limit : float, optional
        Maximum one-way turnover constraint.
    prev_weights : pd.Series, optional
        Previous period weights for turnover constraint.

    Returns
    -------
    pd.Series
        Optimal portfolio weights, shape (N,).
    """
    try:
        import cvxpy as cp
    except ImportError:
        logger.warning("CVXPY not installed. Falling back to equal-weight quintile.")
        return quintile_weights(alpha)

    common = alpha.dropna().index.intersection(cov_matrix.index)
    if len(common) < 5:
        return pd.Series(0.0, index=alpha.index)

    a = alpha.loc[common].values.astype(float)
    S = cov_matrix.loc[common, common].values.astype(float)

    n = len(common)
    w = cp.Variable(n)

    objective = cp.Maximize(a @ w - (risk_aversion / 2) * cp.quad_form(w, S))
    constraints = [
        cp.sum(w) == 0,  # dollar neutral
        w >= min_weight,
        w <= max_weight,
    ]

    if turnover_limit is not None and prev_weights is not None:
        prev = prev_weights.reindex(common).fillna(0.0).values
        constraints.append(
            cp.sum(cp.abs(w - prev)) / 2 <= turnover_limit
        )

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(verbose=False)

    if w.value is None:
        logger.warning("CVXPY optimization failed; returning zero weights.")
        return pd.Series(0.0, index=alpha.index)

    result = pd.Series(0.0, index=alpha.index)
    result.loc[common] = w.value
    return result
