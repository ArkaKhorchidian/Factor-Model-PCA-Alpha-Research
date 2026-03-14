"""
Price and return data fetching with local disk caching.

Data source: yfinance (daily OHLCV bars).
Cache: Parquet files in data/cache/ keyed by ticker list + date range.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_key(tickers: list[str], start: str, end: str, field: str) -> Path:
    """Return a deterministic cache file path."""
    key = "|".join(sorted(tickers)) + f"|{start}|{end}|{field}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{field}_{digest}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    field: str = "Adj Close",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily price data for *tickers* between *start* and *end*.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.
    start : str
        Start date, inclusive, e.g. "2010-01-01".
    end : str
        End date, inclusive, e.g. "2023-12-31".
    field : str
        OHLCV field to return. Default "Adj Close".
    use_cache : bool
        If True, read from / write to local Parquet cache.

    Returns
    -------
    pd.DataFrame
        Shape (T, N) with DatetimeIndex and ticker columns.
        Missing values filled forward then dropped from the front.
    """
    cache_path = _cache_key(tickers, start, end, field.replace(" ", "_"))

    if use_cache and cache_path.exists():
        logger.info("Loading from cache: %s", cache_path.name)
        return pd.read_parquet(cache_path)

    logger.info("Downloading %d tickers from yfinance …", len(tickers))
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[field].copy()
    else:
        prices = raw[[field]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.ffill().dropna(how="all")

    if use_cache:
        prices.to_parquet(cache_path)
        logger.info("Cached to %s", cache_path.name)

    return prices


def fetch_returns(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Return daily simple returns (pct_change) for *tickers*.

    Calls fetch_prices internally; result is (T, N) with the first row dropped.
    """
    prices = fetch_prices(tickers, start, end, field="Adj Close", use_cache=use_cache)
    returns = prices.pct_change().iloc[1:]
    return returns.dropna(how="all")


def fetch_market_caps(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Proxy daily market cap as shares_outstanding × adjusted_close.

    yfinance does not provide historical shares outstanding, so this is an
    approximation using current shares × daily price.  It is sufficient for
    size ranking but not for precise book-to-market ratios.
    """
    prices = fetch_prices(tickers, start, end, use_cache=use_cache)
    mcaps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding", None)
            if shares and ticker in prices.columns:
                mcaps[ticker] = prices[ticker] * shares
        except Exception:
            pass
    if not mcaps:
        return pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    return pd.DataFrame(mcaps).reindex(columns=tickers)


def load_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Load the daily risk-free rate (3-month T-bill) as a decimal daily rate.

    Uses the Fama-French RF factor from pandas-datareader when available,
    falling back to fetching ^IRX from yfinance.
    """
    try:
        import pandas_datareader.data as web

        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)
        rf = ff[0]["RF"] / 100  # convert percentage to decimal
        rf.index = pd.to_datetime(rf.index)
        return rf.rename("rf")
    except Exception:
        logger.warning("pandas-datareader unavailable; using ^IRX proxy for RF.")
        irx = fetch_prices(["^IRX"], start, end, use_cache=True)["^IRX"]
        return (irx / 100 / 252).rename("rf")
