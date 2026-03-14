"""
Barra-style return attribution.

Decomposes realized portfolio returns into:
  1. Factor contributions (market beta, FF factors).
  2. Residual alpha (what remains after factor adjustment).

If the residual alpha is near zero, that is an honest, credible result —
not a failure.  The attribution should be reported transparently.

Reference
---------
Barra Risk Model Handbook (2007), Ch. 4 — Attribution methodology.
Grinold & Kahn (1999), Ch. 7 — Portfolio alpha and factor exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def time_series_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
    add_constant: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Regress excess portfolio returns on factor returns.

    Model: R_p - RF = α + Σ β_k × F_k + ε

    Parameters
    ----------
    portfolio_returns : pd.Series
        Strategy returns, shape (T,).
    factor_returns : pd.DataFrame
        Factor returns (e.g., FF5), shape (T, K).
    risk_free : pd.Series or float
        Risk-free rate.
    add_constant : bool
        If True, estimate alpha (intercept).

    Returns
    -------
    statsmodels RegressionResults object.
    """
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    r_p = portfolio_returns.loc[common_idx]
    f = factor_returns.loc[common_idx]

    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(common_idx).fillna(0.0)
    else:
        rf = risk_free

    y = r_p - rf
    X = sm.add_constant(f) if add_constant else f

    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def attribution_table(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """
    Full attribution table: alpha, betas, t-stats, R², information ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
    factor_returns : pd.DataFrame
        Columns should be named (e.g., "Mkt-RF", "SMB", "HML", "RMW", "CMA").
    risk_free : pd.Series or float

    Returns
    -------
    pd.DataFrame
        Index: "alpha", then factor names.
        Columns: coefficient, t_stat, p_value, contribution_ann.
    """
    result = time_series_regression(portfolio_returns, factor_returns, risk_free)

    periods_per_year = 252
    rows = []

    for name, coef, tstat, pval in zip(
        result.params.index, result.params, result.tvalues, result.pvalues
    ):
        label = name if name != "const" else "alpha"
        ann_coef = coef * periods_per_year if label == "alpha" else coef
        rows.append(
            {
                "term": label,
                "coefficient": coef,
                "annualized": ann_coef,
                "t_stat": tstat,
                "p_value": pval,
                "significant": pval < 0.05,
            }
        )

    df = pd.DataFrame(rows).set_index("term")

    # Add summary stats
    df.attrs["r_squared"] = result.rsquared
    df.attrs["adj_r_squared"] = result.rsquared_adj
    df.attrs["n_obs"] = result.nobs

    return df


def factor_contribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """
    Decompose realized returns into factor contributions and residual alpha.

    Contribution of factor k = β_k × F_k (daily time series).
    Residual = R_p - RF - Σ (β_k × F_k).

    Returns
    -------
    pd.DataFrame
        Columns: one per factor + "alpha" (residual) + "total_explained".
        Shape (T, K+2).
    """
    reg = time_series_regression(portfolio_returns, factor_returns, risk_free)

    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    f = factor_returns.loc[common_idx]

    contributions = pd.DataFrame(index=common_idx)

    for factor_name in factor_returns.columns:
        if factor_name in reg.params.index:
            contributions[factor_name] = reg.params[factor_name] * f[factor_name]

    contributions["total_explained"] = contributions.sum(axis=1)

    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(common_idx).fillna(0.0)
    else:
        rf = risk_free

    rp = portfolio_returns.reindex(common_idx)
    contributions["alpha"] = rp - rf - contributions["total_explained"]
    if "const" in reg.params.index:
        # alpha intercept is constant; residual is variation around it
        contributions["alpha"] = rp - rf - contributions["total_explained"]

    return contributions


def rolling_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: pd.Series | float = 0.0,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling 252-day attribution: track whether alpha or factor exposures
    drive returns over time.

    Returns
    -------
    pd.DataFrame
        Rolling betas and alpha, shape (T, K+1).
    """
    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    rp = portfolio_returns.reindex(common_idx)
    f = factor_returns.reindex(common_idx)
    factor_cols = f.columns.tolist()

    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(common_idx).fillna(0.0)
    else:
        rf = risk_free

    y = rp - rf
    records = {}

    for t in range(window, len(common_idx)):
        slice_y = y.iloc[t - window : t]
        slice_f = f.iloc[t - window : t]
        X = sm.add_constant(slice_f)
        try:
            res = sm.OLS(slice_y, X, missing="drop").fit()
            row = {"alpha_daily": res.params.get("const", np.nan)}
            for fc in factor_cols:
                row[f"beta_{fc}"] = res.params.get(fc, np.nan)
        except Exception:
            row = {"alpha_daily": np.nan}
            for fc in factor_cols:
                row[f"beta_{fc}"] = np.nan
        records[common_idx[t]] = row

    return pd.DataFrame(records).T
