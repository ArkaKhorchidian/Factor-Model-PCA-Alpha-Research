"""
Fama-French factor data retrieval via pandas-datareader.

Provides daily and monthly FF3 / FF5 factor returns directly from
Kenneth French's Data Library.  These are used as benchmarks to validate
hand-constructed factors and as controls in return attribution.

Reference
---------
French, K.R.  Data Library.  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
import pandas_datareader.data as web

logger = logging.getLogger(__name__)

# Mapping: frequency tag → pandas-datareader dataset name
_FF3_DAILY = "F-F_Research_Data_Factors_daily"
_FF5_DAILY = "F-F_Research_Data_5_Factors_2x3_daily"
_FF3_MONTHLY = "F-F_Research_Data_Factors"
_FF5_MONTHLY = "F-F_Research_Data_5_Factors_2x3"
_MOM_DAILY = "F-F_Momentum_Factor_daily"
_MOM_MONTHLY = "F-F_Momentum_Factor"


def _fetch(dataset: str, start: str, end: str) -> pd.DataFrame:
    """Raw fetch from Fama-French data library; converts pct to decimal."""
    df: pd.DataFrame = web.DataReader(dataset, "famafrench", start, end)[0]
    df.index = pd.to_datetime(df.index)
    df = df / 100  # percentage → decimal
    return df.loc[start:end]


def load_ff3(
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Load Fama-French 3-factor data (Mkt-RF, SMB, HML) plus RF.

    Reference
    ---------
    Fama, E.F. & French, K.R. (1993). "Common risk factors in the returns
    on stocks and bonds." *J. Financial Economics*, 33(1), 3–56.

    Parameters
    ----------
    start, end : str
        Date range in ISO format.
    frequency : {"daily", "monthly"}

    Returns
    -------
    pd.DataFrame
        Columns: Mkt-RF, SMB, HML, RF.  Values are decimal returns.
    """
    dataset = _FF3_DAILY if frequency == "daily" else _FF3_MONTHLY
    return _fetch(dataset, start, end)


def load_ff5(
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Load Fama-French 5-factor data (Mkt-RF, SMB, HML, RMW, CMA) plus RF.

    Reference
    ---------
    Fama, E.F. & French, K.R. (2015). "A five-factor asset pricing model."
    *J. Financial Economics*, 116(1), 1–22.

    Parameters
    ----------
    start, end : str
        Date range in ISO format.
    frequency : {"daily", "monthly"}

    Returns
    -------
    pd.DataFrame
        Columns: Mkt-RF, SMB, HML, RMW, CMA, RF.  Values are decimal returns.
    """
    dataset = _FF5_DAILY if frequency == "daily" else _FF5_MONTHLY
    return _fetch(dataset, start, end)


def load_momentum(
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    frequency: str = "daily",
) -> pd.Series:
    """
    Load Fama-French Momentum (UMD/WML) factor.

    Reference
    ---------
    Jegadeesh, N. & Titman, S. (1993). "Returns to buying winners and
    selling losers." *Journal of Finance*, 48(1), 65–91.

    Returns
    -------
    pd.Series  named "Mom".
    """
    dataset = _MOM_DAILY if frequency == "daily" else _MOM_MONTHLY
    df = _fetch(dataset, start, end)
    # Column name varies slightly — take first non-RF column
    col = [c for c in df.columns if "Mom" in c or "WML" in c or "UMD" in c]
    if col:
        return df[col[0]].rename("Mom")
    return df.iloc[:, 0].rename("Mom")


def load_ff5_plus_mom(
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Convenience: FF5 factors joined with Momentum factor.

    Returns
    -------
    pd.DataFrame
        Columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF.
    """
    ff5 = load_ff5(start, end, frequency)
    mom = load_momentum(start, end, frequency)
    combined = ff5.join(mom, how="inner")
    return combined


def factor_names_ff3() -> list[str]:
    return ["Mkt-RF", "SMB", "HML"]


def factor_names_ff5() -> list[str]:
    return ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def factor_names_ff6() -> list[str]:
    return ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
