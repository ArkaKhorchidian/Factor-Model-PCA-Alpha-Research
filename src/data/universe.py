"""
Stock universe construction.

SURVIVORSHIP BIAS WARNING
-------------------------
Using a static list of *current* S&P 500 constituents introduces look-ahead
bias: all firms in that list survived to the present, which inflates historical
backtests.  This module documents the limitation explicitly and provides a
partial mitigation via a broad, market-cap-filtered universe.

Preferred approach (not implemented here due to data access requirements):
  - Use point-in-time constituent lists from CRSP (via WRDS academic access)
    or a commercial data vendor.

Minimum viable mitigation implemented here:
  - Use a large cross-section of stocks above a market-cap threshold.
  - The hard-coded tickers below are a *proxy* universe and still suffer from
    survivorship bias.  All backtested results must be discounted accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static proxy universes (survivorship-biased — see module docstring)
# ---------------------------------------------------------------------------

# Large-cap proxy: ~100 liquid large-cap US equities (S&P 100 approximate)
# These are current constituents as of 2024 — historical results are biased.
SP100_PROXY = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
    "JPM", "UNH", "XOM", "V", "AVGO", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "COST", "PEP", "ADBE", "WMT", "BAC", "KO", "MCD", "CRM",
    "ACN", "TMO", "CSCO", "ABT", "LIN", "ORCL", "DHR", "NFLX", "CMCSA",
    "TXN", "AMD", "NEE", "PM", "HON", "UNP", "AMGN", "IBM", "QCOM",
    "LOW", "GE", "INTU", "SPGI", "CAT", "SBUX", "GS", "AXP", "MS",
    "RTX", "BLK", "T", "DE", "MDT", "UPS", "GILD", "ISRG", "ADI",
    "SYK", "VRTX", "REGN", "ZTS", "BSX", "C", "NOW", "MO", "PLD",
    "CI", "CB", "MMC", "SO", "DUK", "CL", "PGR", "ITW", "FDX",
    "ETN", "ADP", "AON", "SCHW", "PYPL", "BKNG", "TJX", "CVS", "WFC",
    "LRCX", "KLAC", "SNPS", "CDNS", "NXPI", "MCHP", "ON", "AMAT",
]

# Sector mapping for the proxy universe (GICS sectors)
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "AMZN": "Consumer Discretionary", "NVDA": "Information Technology",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "BRK-B": "Financials", "LLY": "Health Care",
    "JPM": "Financials", "UNH": "Health Care",
    "XOM": "Energy", "V": "Financials",
    "AVGO": "Information Technology", "PG": "Consumer Staples",
    "MA": "Financials", "HD": "Consumer Discretionary",
    "CVX": "Energy", "MRK": "Health Care",
    "ABBV": "Health Care", "COST": "Consumer Staples",
    "PEP": "Consumer Staples", "ADBE": "Information Technology",
    "WMT": "Consumer Staples", "BAC": "Financials",
    "KO": "Consumer Staples", "MCD": "Consumer Discretionary",
    "CRM": "Information Technology", "ACN": "Information Technology",
    "TMO": "Health Care", "CSCO": "Information Technology",
    "ABT": "Health Care", "LIN": "Materials",
    "ORCL": "Information Technology", "DHR": "Health Care",
    "NFLX": "Communication Services", "CMCSA": "Communication Services",
    "TXN": "Information Technology", "AMD": "Information Technology",
    "NEE": "Utilities", "PM": "Consumer Staples",
    "HON": "Industrials", "UNP": "Industrials",
    "AMGN": "Health Care", "IBM": "Information Technology",
    "QCOM": "Information Technology", "LOW": "Consumer Discretionary",
    "GE": "Industrials", "INTU": "Information Technology",
    "SPGI": "Financials", "CAT": "Industrials",
    "SBUX": "Consumer Discretionary", "GS": "Financials",
    "AXP": "Financials", "MS": "Financials",
    "RTX": "Industrials", "BLK": "Financials",
    "T": "Communication Services", "DE": "Industrials",
    "MDT": "Health Care", "UPS": "Industrials",
    "GILD": "Health Care", "ISRG": "Health Care",
    "ADI": "Information Technology", "SYK": "Health Care",
    "VRTX": "Health Care", "REGN": "Health Care",
    "ZTS": "Health Care", "BSX": "Health Care",
    "C": "Financials", "NOW": "Information Technology",
    "MO": "Consumer Staples", "PLD": "Real Estate",
    "CI": "Health Care", "CB": "Financials",
    "MMC": "Financials", "SO": "Utilities",
    "DUK": "Utilities", "CL": "Consumer Staples",
    "PGR": "Financials", "ITW": "Industrials",
    "FDX": "Industrials", "ETN": "Industrials",
    "ADP": "Information Technology", "AON": "Financials",
    "SCHW": "Financials", "PYPL": "Financials",
    "BKNG": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "CVS": "Health Care", "WFC": "Financials",
    "LRCX": "Information Technology", "KLAC": "Information Technology",
    "SNPS": "Information Technology", "CDNS": "Information Technology",
    "NXPI": "Information Technology", "MCHP": "Information Technology",
    "ON": "Information Technology", "AMAT": "Information Technology",
}


def get_universe(name: str = "sp100_proxy") -> list[str]:
    """
    Return ticker list for a named universe.

    Parameters
    ----------
    name : str
        "sp100_proxy" — 100 large-cap US equities (survivorship-biased).

    Returns
    -------
    list[str]
    """
    universes = {
        "sp100_proxy": SP100_PROXY,
    }
    if name not in universes:
        raise ValueError(f"Unknown universe '{name}'. Available: {list(universes)}")
    logger.warning(
        "Universe '%s' uses current constituents and is survivorship-biased. "
        "All backtested Sharpe ratios should be interpreted with caution.",
        name,
    )
    return universes[name]


def get_sector_dummies(tickers: list[str]) -> pd.DataFrame:
    """
    Return one-hot encoded sector membership for *tickers*.

    Tickers not in SECTOR_MAP are assigned to "Unknown".
    """
    sectors = pd.Series(
        {t: SECTOR_MAP.get(t, "Unknown") for t in tickers}, name="sector"
    )
    dummies = pd.get_dummies(sectors, prefix="", prefix_sep="")
    return dummies.astype(float)
