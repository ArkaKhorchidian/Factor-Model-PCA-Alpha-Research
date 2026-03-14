"""
Utility functions: winsorization, z-scoring, neutralization helpers.

All cross-sectional operations work on a single date's cross-section (pd.Series)
and are designed to be applied via groupby or rolling apply in calling code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Cross-sectional standardisation
# ---------------------------------------------------------------------------


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip a cross-sectional series to the [lower, upper] quantile range.

    Parameters
    ----------
    series : pd.Series
        Raw signal values for a single cross-section.
    lower : float
        Lower quantile bound (default 1st percentile).
    upper : float
        Upper quantile bound (default 99th percentile).

    Returns
    -------
    pd.Series
        Winsorized series with the same index.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def zscore(series: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score: (x - mean) / std.

    Returns NaN-safe result; std=0 yields NaN.
    """
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma < 1e-10 or np.isnan(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def rank_zscore(series: pd.Series) -> pd.Series:
    """
    Convert a raw signal to a rank-based z-score.

    Rank → [0,1] quantile → normal quantile via inverse CDF.
    More robust to outliers than direct z-score.
    """
    from scipy.stats import norm

    pct = series.rank(pct=True)
    # Clip to avoid ±inf from norm.ppf(0) / norm.ppf(1)
    pct = pct.clip(0.01, 0.99)
    return pd.Series(norm.ppf(pct.values), index=series.index)


def winsorize_and_zscore(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Convenience: winsorize then z-score."""
    return zscore(winsorize(series, lower=lower, upper=upper))


# ---------------------------------------------------------------------------
# Factor neutralization (residualization)
# ---------------------------------------------------------------------------


def neutralize(signal: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """
    Residualize *signal* against *factors* via cross-sectional OLS.

    Returns the component of signal unexplained by the provided factors,
    i.e. the pure alpha orthogonal to the factor exposures.

    Reference
    ---------
    Grinold & Kahn (1999), *Active Portfolio Management*, Ch. 3.

    Parameters
    ----------
    signal : pd.Series
        Cross-sectional signal, shape (N,).
    factors : pd.DataFrame
        Factor exposures, shape (N, K).  Must share the same index as signal.

    Returns
    -------
    pd.Series
        Residualized signal with same index as input.
    """
    common = signal.index.intersection(factors.index)
    if len(common) < factors.shape[1] + 2:
        return pd.Series(np.nan, index=signal.index)

    y = signal.loc[common].values
    X = factors.loc[common].values

    reg = LinearRegression(fit_intercept=True).fit(X, y)
    residuals = y - reg.predict(X)
    result = pd.Series(np.nan, index=signal.index)
    result.loc[common] = residuals
    return result


def market_cap_neutralize(
    signal: pd.Series, log_mcap: pd.Series
) -> pd.Series:
    """
    Remove the cross-sectional relationship between signal and log(market cap).

    Equivalent to regressing signal on log_mcap and returning residuals.
    """
    factors = log_mcap.rename("log_mcap").to_frame()
    return neutralize(signal, factors)


def sector_neutralize(signal: pd.Series, sector_dummies: pd.DataFrame) -> pd.Series:
    """
    Demean the signal within each GICS sector.

    Parameters
    ----------
    signal : pd.Series
        Cross-sectional signal.
    sector_dummies : pd.DataFrame
        One-hot encoded sector membership, shape (N, S).

    Returns
    -------
    pd.Series
        Sector-demeaned signal.
    """
    return neutralize(signal, sector_dummies)


# ---------------------------------------------------------------------------
# Return utilities
# ---------------------------------------------------------------------------


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns: (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change()


def rolling_cov(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Compute trailing rolling covariance matrix using the last *window* periods.

    Returns the covariance matrix as of the most recent date (end of the window).
    """
    return returns.iloc[-window:].cov()


def align_dates(*series: pd.Series | pd.DataFrame) -> list:
    """
    Inner-join multiple series/frames on their DatetimeIndex.

    Returns a list of aligned objects in the same order as input.
    """
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    return [s.loc[common] for s in series]
