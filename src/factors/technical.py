"""
Technical / price-based signals: momentum, reversal, volatility, liquidity.

All signals are computed cross-sectionally and returned as DataFrames with
the same (T, N) shape as the input returns DataFrame.  They are raw signals
before winsorization or z-scoring — apply src.utils.winsorize_and_zscore
before use in a portfolio.

References
----------
Jegadeesh & Titman (1993) — 12-1 month momentum.
De Bondt & Thaler (1985) — long-term reversal.
Ang, Hodrick, Xing & Zhang (2006) — idiosyncratic volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional momentum signals
# ---------------------------------------------------------------------------


def momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 month cross-sectional momentum signal.

    Signal at date t = cumulative return over [t-252, t-21] trading days.
    The most recent month (≈21 trading days) is skipped to avoid contamination
    by short-term reversal.

    Reference: Jegadeesh & Titman (1993), Eq. (1).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, shape (T, N).

    Returns
    -------
    pd.DataFrame
        Momentum scores, shape (T, N). NaN where insufficient history.
    """
    long_window = 252   # 12 months
    skip_window = 21    # 1 month skip

    prices = (1 + returns).cumprod()
    # Cumulative return from 252 days ago to 21 days ago
    mom = (
        prices.shift(skip_window) / prices.shift(long_window) - 1
    )
    return mom.rename(columns={c: c for c in mom.columns})


def short_term_reversal(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Short-term (1-month) reversal signal.

    Signal at date t = -1 × return over the past *window* trading days.
    Negative sign: past losers are expected to outperform (reversal).

    Reference: Jegadeesh (1990), "Evidence of Predictable Behavior of Security Returns."

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, shape (T, N).
    window : int
        Look-back window in trading days (default 21 ≈ 1 month).

    Returns
    -------
    pd.DataFrame
        Reversal scores (negative of rolling return), shape (T, N).
    """
    rolling_return = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    return -rolling_return


def long_term_reversal(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Long-term (3-5 year) reversal signal.

    Signal at date t = -1 × cumulative return over [t-1260, t-252] trading days.
    Captures De Bondt & Thaler (1985) overreaction reversal.

    Reference: De Bondt, W. & Thaler, R. (1985). "Does the Stock Market Overreact?"
    *J. Finance*, 40(3), 793–805.
    """
    prices = (1 + returns).cumprod()
    lt_return = prices.shift(252) / prices.shift(1260) - 1
    return -lt_return


# ---------------------------------------------------------------------------
# Volatility signals
# ---------------------------------------------------------------------------


def realized_volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Trailing annualised realised volatility (rolling std × √252).

    Low-volatility anomaly: lower-vol stocks historically outperform on a
    risk-adjusted basis (Ang et al. 2006).

    Parameters
    ----------
    returns : pd.DataFrame
    window : int
        Rolling window in trading days (default 63 ≈ 3 months).

    Returns
    -------
    pd.DataFrame
        Annualised volatility, shape (T, N).
    """
    return returns.rolling(window).std() * np.sqrt(252)


def idiosyncratic_volatility(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling idiosyncratic volatility: std of residuals from market regression.

    Reference: Ang, Hodrick, Xing & Zhang (2006). "The Cross-Section of
    Volatility and Expected Returns." *J. Finance*, 61(1), 259–299.  Eq. (1).

    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns, shape (T, N).
    market_returns : pd.Series
        Market (Mkt-RF or benchmark) returns, shape (T,).
    window : int
        Rolling window.

    Returns
    -------
    pd.DataFrame
        Idiosyncratic vol, shape (T, N).
    """
    from sklearn.linear_model import LinearRegression

    ivol = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for end in range(window, len(returns) + 1):
        start = end - window
        ret_slice = returns.iloc[start:end]
        mkt_slice = market_returns.iloc[start:end].values.reshape(-1, 1)
        date = returns.index[end - 1]

        residuals = {}
        for col in returns.columns:
            y = ret_slice[col].values
            mask = ~np.isnan(y) & ~np.isnan(mkt_slice.ravel())
            if mask.sum() < 20:
                residuals[col] = np.nan
                continue
            reg = LinearRegression().fit(mkt_slice[mask], y[mask])
            res = y[mask] - reg.predict(mkt_slice[mask])
            residuals[col] = res.std() * np.sqrt(252)
        ivol.loc[date] = pd.Series(residuals)

    return ivol


# ---------------------------------------------------------------------------
# Liquidity / volume signals
# ---------------------------------------------------------------------------


def amihud_illiquidity(
    returns: pd.DataFrame,
    volume_usd: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio: |R| / DollarVolume, rolled over window.

    Higher values = less liquid.  Often used as a control variable.

    Reference: Amihud, Y. (2002). "Illiquidity and Stock Returns."
    *J. Financial Markets*, 5(1), 31–56. Eq. (1).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns.
    volume_usd : pd.DataFrame
        Daily dollar trading volume (price × volume), same shape as returns.
    window : int
        Rolling window.

    Returns
    -------
    pd.DataFrame
        Amihud illiquidity ratio, shape (T, N).
    """
    ratio = returns.abs() / volume_usd.replace(0, np.nan)
    return ratio.rolling(window).mean()


def dollar_volume(prices: pd.DataFrame, volume_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dollar volume = price × shares traded.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    volume_shares : pd.DataFrame
        Number of shares traded (from yfinance Volume column).

    Returns
    -------
    pd.DataFrame
        Dollar trading volume.
    """
    return prices * volume_shares


# ---------------------------------------------------------------------------
# Composite helpers
# ---------------------------------------------------------------------------


def all_technical_signals(
    returns: pd.DataFrame,
    market_returns: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all technical signals and return as a named dict.

    Parameters
    ----------
    returns : pd.DataFrame
    market_returns : pd.Series, optional
        Required for idiosyncratic vol.

    Returns
    -------
    dict mapping signal name → DataFrame of shape (T, N).
    """
    signals: dict[str, pd.DataFrame] = {
        "momentum_12_1": momentum_12_1(returns),
        "reversal_1m": short_term_reversal(returns),
        "long_term_reversal": long_term_reversal(returns),
        "realized_vol": realized_volatility(returns),
    }
    if market_returns is not None:
        signals["idiosyncratic_vol"] = idiosyncratic_volatility(
            returns, market_returns
        )
    return signals
