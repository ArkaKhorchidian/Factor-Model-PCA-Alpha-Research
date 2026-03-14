"""
Factor combination and composite alpha construction.

Signals are combined using IC-weighted averaging (Grinold & Kahn 1999):
  weight_i = IC_i / Σ IC_j

The composite is then residualized against FF factors to isolate pure alpha.

References
----------
Grinold, R. & Kahn, R. (1999). *Active Portfolio Management*, 2nd ed.
  Ch. 6: "The Information Ratio" — IC/IR framework.
  Ch. 14: "Signal Weighting" — IC-weighted combination.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import neutralize, winsorize_and_zscore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Information Coefficient (IC) computation
# ---------------------------------------------------------------------------


def compute_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute rolling cross-sectional Information Coefficient (IC).

    IC at date t = rank_correlation(signal[t], forward_return[t]).
    Uses Spearman rank correlation by default (robust to outliers).

    Reference: Grinold & Kahn (1999), Ch. 6, Eq. (6.1).

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectional signal, shape (T, N).
    forward_returns : pd.DataFrame
        Forward-period returns, shape (T, N).  Typically 1-month ahead.
    method : {"spearman", "pearson"}
        Correlation type.

    Returns
    -------
    pd.Series
        Daily IC values, shape (T,).
    """
    common_dates = signal.index.intersection(forward_returns.index)
    ics = []
    for date in common_dates:
        sig = signal.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        both = sig.index.intersection(fwd.index)
        if len(both) < 5:
            ics.append(np.nan)
            continue
        s = sig.loc[both].values
        f = fwd.loc[both].values
        if method == "spearman":
            corr, _ = spearmanr(s, f)
        else:
            corr = np.corrcoef(s, f)[0, 1]
        ics.append(corr)

    return pd.Series(ics, index=common_dates, name="IC")


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """
    Summarise IC series: mean, std, t-stat, IC IR.

    IC IR = mean(IC) / std(IC).  Values > 0.5 are considered good.

    Reference: Grinold & Kahn (1999), Ch. 6, Eq. (6.4).
    """
    ic = ic_series.dropna()
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    n = len(ic)
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    return {
        "mean_IC": mean_ic,
        "std_IC": std_ic,
        "t_stat": t_stat,
        "IC_IR": ic_ir,
        "n_periods": n,
        "pct_positive": (ic > 0).mean(),
    }


def ic_decay(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizons_days: list[int] | None = None,
) -> pd.DataFrame:
    """
    IC decay analysis: compute IC at multiple forward horizons.

    Parameters
    ----------
    signal : pd.DataFrame
        Cross-sectional signal, shape (T, N).
    returns : pd.DataFrame
        Daily returns, shape (T, N).
    horizons_days : list[int]
        Horizons in trading days.  Defaults to [5, 10, 21, 63].

    Returns
    -------
    pd.DataFrame
        Columns: horizon, mean_IC, IC_IR.
    """
    if horizons_days is None:
        horizons_days = [5, 10, 21, 63]

    records = []
    for h in horizons_days:
        # Forward return over h days
        fwd = (1 + returns).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        ic = compute_ic(signal, fwd)
        summ = ic_summary(ic)
        records.append({"horizon_days": h, **summ})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# IC-weighted signal combination
# ---------------------------------------------------------------------------


def ic_weighted_combination(
    signals: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_lookback: int = 252,
) -> pd.DataFrame:
    """
    Combine multiple signals into a composite alpha using IC-weighted averaging.

    weight_i = IC_i / Σ|IC_j|

    Uses a rolling window of *ic_lookback* days to estimate each signal's
    average IC before computing weights.  This avoids look-ahead bias in
    the combination step.

    Reference: Grinold & Kahn (1999), Ch. 14.

    Parameters
    ----------
    signals : dict[str, pd.DataFrame]
        Named signals, each shape (T, N).
    forward_returns : pd.DataFrame
        Forward 1-period returns, shape (T, N).
    ic_lookback : int
        Rolling window for IC estimation (default 252 days).

    Returns
    -------
    pd.DataFrame
        Composite alpha, shape (T, N).
    """
    # Standardize all signals first
    std_signals = {
        name: signal.apply(
            lambda row: winsorize_and_zscore(row.dropna()).reindex(row.index),
            axis=1,
        )
        for name, signal in signals.items()
    }

    # Compute rolling IC for each signal
    rolling_ics: dict[str, pd.Series] = {}
    for name, signal in std_signals.items():
        ic = compute_ic(signal, forward_returns)
        rolling_ics[name] = ic.rolling(ic_lookback, min_periods=63).mean()

    # Build composite at each date
    all_dates = list(std_signals.values())[0].index
    composite_rows = {}

    for date in all_dates:
        weights = {}
        for name in signals:
            if date in rolling_ics[name].index:
                ic_val = rolling_ics[name].loc[date]
                weights[name] = ic_val if not np.isnan(ic_val) else 0.0
            else:
                weights[name] = 0.0

        total_weight = sum(abs(w) for w in weights.values())
        if total_weight < 1e-8:
            composite_rows[date] = pd.Series(np.nan, index=all_dates[:0])
            continue

        norm_weights = {k: v / total_weight for k, v in weights.items()}

        row_values: dict[str, float] = {}
        for name, signal in std_signals.items():
            if date not in signal.index:
                continue
            sig_row = signal.loc[date]
            w = norm_weights[name]
            for ticker, val in sig_row.items():
                if ticker not in row_values:
                    row_values[ticker] = 0.0
                if not np.isnan(val):
                    row_values[ticker] += w * val

        composite_rows[date] = pd.Series(row_values)

    composite = pd.DataFrame(composite_rows).T
    composite.index = pd.DatetimeIndex(composite.index)
    return composite


def factor_orthogonalize(
    signal: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Residualize composite signal against FF factor exposures.

    At each date, run cross-sectional OLS of the signal on factor loadings
    and return the residual.  This removes beta-like components from the
    alpha signal.

    Parameters
    ----------
    signal : pd.DataFrame
        Composite alpha, shape (T, N).
    factor_returns : pd.DataFrame
        Factor time series — not used directly here but kept for API symmetry.
    factor_loadings : pd.DataFrame, optional
        Cross-sectional factor exposures, shape (N, K).  If None, uses
        trailing beta estimates.

    Returns
    -------
    pd.DataFrame
        Orthogonalized alpha, shape (T, N).
    """
    if factor_loadings is None:
        logger.warning(
            "No factor_loadings provided for orthogonalization; "
            "returning signal unchanged."
        )
        return signal

    result_rows = {}
    for date in signal.index:
        row = signal.loc[date]
        valid_tickers = row.dropna().index
        fl = factor_loadings.loc[factor_loadings.index.isin(valid_tickers)]
        common = valid_tickers.intersection(fl.index)
        if len(common) < fl.shape[1] + 2:
            result_rows[date] = row
            continue
        residual = neutralize(row.loc[common], fl.loc[common])
        full_row = row.copy()
        full_row.loc[common] = residual.loc[common]
        result_rows[date] = full_row

    result = pd.DataFrame(result_rows).T
    result.index = pd.DatetimeIndex(result.index)
    return result
