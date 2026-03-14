"""
Information Coefficient (IC) analysis: decay, stability, and sector attribution.

IC is the single most important diagnostic for a quantitative signal.
A strategy with mean IC=0.04 and IC_IR=0.8 is more credible than one with
IC=0.12 and IC_IR=0.3 (the latter is noisy/overfit).

Reference
---------
Grinold, R. & Kahn, R. (1999). *Active Portfolio Management*, Ch. 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


def rolling_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """
    Compute trailing rolling-mean IC over *window* periods.

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectional signal, shape (T, N).
    forward_returns : pd.DataFrame
        Next-period returns, shape (T, N).
    window : int
        Rolling window for smoothing.

    Returns
    -------
    pd.Series
        Smoothed IC time series.
    """
    from src.factors.combine import compute_ic
    raw_ic = compute_ic(signal, forward_returns)
    return raw_ic.rolling(window, min_periods=21).mean().rename("rolling_IC")


def ic_by_sector(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Compute IC separately for each GICS sector.

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectional signal, shape (T, N).
    forward_returns : pd.DataFrame
        Forward returns, shape (T, N).
    sector_map : dict[str, str]
        Mapping ticker → sector name.

    Returns
    -------
    pd.DataFrame
        Columns: sector, mean_IC, std_IC, IC_IR, n_dates.
    """
    sectors = pd.Series(sector_map)
    unique_sectors = sectors.unique()
    records = []

    for sector in unique_sectors:
        tickers = sectors[sectors == sector].index.tolist()
        tickers_in_sig = [t for t in tickers if t in signal.columns]
        if not tickers_in_sig:
            continue
        sig_sub = signal[tickers_in_sig]
        fwd_sub = forward_returns.reindex(columns=tickers_in_sig)
        from src.factors.combine import compute_ic, ic_summary
        ic = compute_ic(sig_sub, fwd_sub)
        summ = ic_summary(ic)
        records.append({"sector": sector, **summ})

    return pd.DataFrame(records).set_index("sector")


def ic_autocorrelation(ic_series: pd.Series, max_lag: int = 12) -> pd.Series:
    """
    Compute IC series autocorrelation at lags 1..max_lag.

    High autocorrelation suggests signal persistence (or look-ahead contamination).

    Returns
    -------
    pd.Series
        Autocorrelation values, index = lag.
    """
    ic = ic_series.dropna()
    acf_vals = {lag: ic.autocorr(lag=lag) for lag in range(1, max_lag + 1)}
    return pd.Series(acf_vals, name="IC_autocorrelation")


def hit_rate_by_quintile(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """
    For each quintile of the signal, compute the fraction of times forward
    return was positive (hit rate).

    Monotone hit rates from Q1 < Q2 < ... < Q5 confirm signal directionality.

    Returns
    -------
    pd.DataFrame
        Columns: quintile, mean_forward_return, hit_rate, n_obs.
    """
    records = []
    common_dates = signal.index.intersection(forward_returns.index)

    all_sig = []
    all_fwd = []
    for date in common_dates:
        sig_row = signal.loc[date].dropna()
        fwd_row = forward_returns.loc[date].dropna()
        common_tickers = sig_row.index.intersection(fwd_row.index)
        if len(common_tickers) < n_quintiles:
            continue
        s = sig_row.loc[common_tickers]
        f = fwd_row.loc[common_tickers]
        q = pd.qcut(s.rank(method="first"), n_quintiles, labels=False) + 1
        for q_val in range(1, n_quintiles + 1):
            in_q = q[q == q_val].index
            all_sig.extend([(q_val, ticker) for ticker in in_q])
            all_fwd.extend(f.loc[in_q].tolist())

    if not all_sig:
        return pd.DataFrame()

    qdf = pd.DataFrame(all_sig, columns=["quintile", "ticker"])
    qdf["forward_return"] = all_fwd
    for q_val, group in qdf.groupby("quintile"):
        fwd = group["forward_return"]
        records.append(
            {
                "quintile": q_val,
                "mean_forward_return": fwd.mean(),
                "hit_rate": (fwd > 0).mean(),
                "n_obs": len(fwd),
            }
        )

    return pd.DataFrame(records).set_index("quintile")


def ic_t_test(ic_series: pd.Series) -> dict[str, float]:
    """
    One-sample t-test: H0: mean(IC) = 0 vs. H1: mean(IC) ≠ 0.

    Parameters
    ----------
    ic_series : pd.Series

    Returns
    -------
    dict with keys: t_stat, p_value, reject_H0 (at 5% level).
    """
    ic = ic_series.dropna()
    t_stat, p_val = ttest_1samp(ic, popmean=0.0)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "reject_H0_5pct": bool(p_val < 0.05),
        "mean_IC": float(ic.mean()),
        "n": len(ic),
    }
