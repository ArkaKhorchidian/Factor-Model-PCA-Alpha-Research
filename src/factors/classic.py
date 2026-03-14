"""
Classic Fama-French factors: SMB, HML, MOM (12-1), RMW, CMA.

These are constructed from first principles using a cross-section of stock
returns and fundamental data (proxied by market cap and book-to-market
approximations where fundamental data is unavailable).

Note on Data Quality
--------------------
True FF factor construction requires CRSP/Compustat data for book values.
Without those, we proxy B/M with the inverse of trailing P/E or simply
use price-to-book from yfinance.get_info().  The constructed factors will
correlate with but not exactly replicate the published FF series.

References
----------
Fama, E.F. & French, K.R. (1993). Eq. (1)–(4) for SMB and HML construction.
Fama, E.F. & French, K.R. (2015). Eq. (1)–(3) for RMW and CMA.
Jegadeesh & Titman (1993) for momentum.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import winsorize, zscore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMB — Small Minus Big
# ---------------------------------------------------------------------------


def compute_smb(
    returns: pd.DataFrame,
    market_caps: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Construct SMB (Small Minus Big) factor.

    Methodology (Fama & French 1993, p. 8):
      - Rank stocks by market cap at each rebalance date.
      - Small: bottom 50% by market cap.
      - Big:   top 50% by market cap.
      - SMB = equal-weight return(Small) - equal-weight return(Big).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, shape (T, N).
    market_caps : pd.DataFrame
        Market capitalizations (same index/columns as returns).
    rebalance_dates : DatetimeIndex, optional
        Dates at which portfolios are reformed.  Defaults to first trading
        day of each June (annual rebalancing per FF convention).

    Returns
    -------
    pd.Series
        Daily SMB factor returns with DatetimeIndex.
    """
    if rebalance_dates is None:
        # Annual rebalancing at start of July (June-end market caps used)
        rebalance_dates = returns.index[
            returns.index.to_period("M").month == 7
        ][[0]]
        # Actually build monthly rebalancing as a reasonable approximation
        rebalance_dates = returns.resample("MS").first().index

    smb_returns = []
    dates = []

    portfolio_small: list[str] = []
    portfolio_big: list[str] = []
    current_reb = rebalance_dates[0] if len(rebalance_dates) > 0 else returns.index[0]
    reb_iter = iter(rebalance_dates)
    next_reb = next(reb_iter, None)

    for date in returns.index:
        if next_reb is not None and date >= next_reb:
            # Reform portfolios at rebalance date using prior-day market cap
            prev_mcap = market_caps.loc[:date].iloc[-2] if len(market_caps.loc[:date]) > 1 else market_caps.loc[:date].iloc[-1]
            valid = prev_mcap.dropna()
            if len(valid) >= 4:
                median_cap = valid.median()
                portfolio_small = valid[valid < median_cap].index.tolist()
                portfolio_big = valid[valid >= median_cap].index.tolist()
            next_reb = next(reb_iter, None)

        if portfolio_small and portfolio_big:
            r = returns.loc[date]
            ret_small = r[portfolio_small].mean()
            ret_big = r[portfolio_big].mean()
            smb_returns.append(ret_small - ret_big)
        else:
            smb_returns.append(np.nan)
        dates.append(date)

    return pd.Series(smb_returns, index=pd.DatetimeIndex(dates), name="SMB")


# ---------------------------------------------------------------------------
# HML — High Minus Low (book-to-market)
# ---------------------------------------------------------------------------


def compute_hml(
    returns: pd.DataFrame,
    book_to_market: pd.DataFrame,
    market_caps: pd.DataFrame,
) -> pd.Series:
    """
    Construct HML (High Minus Low) factor.

    Methodology (Fama & French 1993, p. 8):
      - Rank by B/M at each rebalance date (size-independent sort).
      - High B/M (value): top 30%.
      - Low  B/M (growth): bottom 30%.
      - HML = equal-weight return(High B/M) - equal-weight return(Low B/M).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, shape (T, N).
    book_to_market : pd.DataFrame
        Book-to-market ratios, shape (T, N).  Can be sparse (e.g., annual).
    market_caps : pd.DataFrame
        Market capitalizations (used for value-weighted variants).

    Returns
    -------
    pd.Series
        Daily HML factor returns.
    """
    # Forward-fill B/M ratios so each trading day has a value
    bm_ffill = book_to_market.reindex(returns.index).ffill()

    hml_list = []
    for date in returns.index:
        bm_today = bm_ffill.loc[date].dropna()
        if len(bm_today) < 6:
            hml_list.append(np.nan)
            continue
        q30 = bm_today.quantile(0.30)
        q70 = bm_today.quantile(0.70)
        high_bm = bm_today[bm_today >= q70].index.tolist()
        low_bm = bm_today[bm_today <= q30].index.tolist()
        r = returns.loc[date]
        ret_high = r[high_bm].mean()
        ret_low = r[low_bm].mean()
        hml_list.append(ret_high - ret_low)

    return pd.Series(hml_list, index=returns.index, name="HML")


# ---------------------------------------------------------------------------
# Momentum — 12-1 month cross-sectional (Jegadeesh & Titman 1993)
# ---------------------------------------------------------------------------


def compute_momentum_factor(
    returns: pd.DataFrame,
    formation_months: int = 12,
    skip_months: int = 1,
) -> pd.Series:
    """
    Construct WML/UMD momentum factor (Winners Minus Losers).

    Methodology (Jegadeesh & Titman 1993, p. 67):
      - Formation period: cumulative return over months [t-12, t-2].
      - Skip month [t-1] to avoid short-term reversal contamination.
      - Long top decile, short bottom decile.
      - Returns equal-weight long minus short monthly return.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns.
    formation_months : int
        Look-back length in months (default 12).
    skip_months : int
        Number of months to skip before current month (default 1).

    Returns
    -------
    pd.Series
        Daily momentum factor returns (held constant within each month).
    """
    # Convert to monthly returns for formation period calculation
    monthly_ret = (1 + returns).resample("ME").prod() - 1

    mom_list = []
    mom_dates = []

    skip_periods = skip_months
    form_periods = formation_months

    for i in range(form_periods + skip_periods, len(monthly_ret)):
        formation_start = i - form_periods - skip_periods
        formation_end = i - skip_periods - 1
        formation = monthly_ret.iloc[formation_start : formation_end + 1]
        cum_ret = (1 + formation).prod() - 1

        valid = cum_ret.dropna()
        if len(valid) < 10:
            continue

        q10 = valid.quantile(0.10)
        q90 = valid.quantile(0.90)
        winners = valid[valid >= q90].index.tolist()
        losers = valid[valid <= q10].index.tolist()

        # Holding period: current month
        holding = monthly_ret.iloc[i]
        ret_w = holding[winners].mean()
        ret_l = holding[losers].mean()
        mom_list.append(ret_w - ret_l)
        mom_dates.append(monthly_ret.index[i])

    monthly_mom = pd.Series(mom_list, index=pd.DatetimeIndex(mom_dates), name="Mom")

    # Broadcast monthly factor to daily frequency
    daily_mom = monthly_mom.reindex(returns.index).ffill()
    return daily_mom
