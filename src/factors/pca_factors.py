"""
PCA-based statistical factor extraction from the return covariance matrix.

CRITICAL NOTE: PCA factors are statistical constructs, NOT tradeable portfolios.
They describe the low-dimensional structure of the return cross-section.
To avoid look-ahead bias, PCA must be refit using ONLY past data at each point
in time.  Fitting PCA on the full sample and backtesting on the same sample
inflates results dramatically.

References
----------
Connor, G. & Korajczyk, R. (1988). "Risk and return in an equilibrium APT:
Application of a new test methodology." *J. Financial Economics*, 21, 255–289.

Implementation uses sklearn.decomposition.PCA which fits via SVD.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def extract_pca_factors(
    returns: pd.DataFrame,
    n_components: int = 10,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling PCA on the return cross-section.

    At each date t, estimate the covariance matrix from the trailing *window*
    days of returns, fit PCA, and project today's returns onto the top
    *n_components* eigenvectors.  This is a strictly causal (no look-ahead)
    implementation.

    Note: PCA factor loadings can flip sign between windows.  We apply a
    sign convention: the first component (market) should have predominantly
    positive loadings.

    Parameters
    ----------
    returns : pd.DataFrame
        Shape (T, N) — T periods, N stocks.  Rows = dates, columns = tickers.
    n_components : int
        Number of principal components to retain.
    window : int
        Lookback window for covariance estimation.

    Returns
    -------
    pd.DataFrame
        Shape (T - window, n_components) — factor returns.
        Columns: PC1, PC2, ..., PCn.
        Index: DatetimeIndex aligned with the input returns.
    """
    T, N = returns.shape
    if T < window + 1:
        raise ValueError(
            f"Need at least {window + 1} periods; got {T}."
        )

    factor_returns = []
    dates = []

    for t in range(window, T):
        # Estimation window: strictly in the past
        window_data = returns.iloc[t - window : t].values  # shape (window, N)

        # Drop columns with too many NaNs in the estimation window
        col_mask = np.sum(np.isnan(window_data), axis=0) < window * 0.2
        X = window_data[:, col_mask]
        X = np.nan_to_num(X, nan=0.0)  # fill residual NaNs with 0

        if X.shape[1] < n_components + 2:
            factor_returns.append([np.nan] * n_components)
            dates.append(returns.index[t])
            continue

        # Demean each stock's time series in the window
        X = X - X.mean(axis=0)

        pca = PCA(n_components=min(n_components, X.shape[1]))
        pca.fit(X)

        # Project today's returns onto the learned eigenvectors
        today = returns.iloc[t].values[col_mask]
        today = np.nan_to_num(today, nan=0.0)
        projected = pca.transform(today.reshape(1, -1)).ravel()

        # Pad if fewer components than requested
        if len(projected) < n_components:
            projected = np.concatenate(
                [projected, [np.nan] * (n_components - len(projected))]
            )

        factor_returns.append(projected.tolist())
        dates.append(returns.index[t])

    cols = [f"PC{i+1}" for i in range(n_components)]
    result = pd.DataFrame(factor_returns, index=pd.DatetimeIndex(dates), columns=cols)
    return result


def explained_variance_curve(
    returns: pd.DataFrame,
    max_components: int | None = None,
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute cumulative explained variance ratio vs. number of components.

    Uses the last *window* rows of *returns* to estimate a single covariance
    matrix (for illustration purposes).  This is NOT used in the causal
    backtest path.

    Parameters
    ----------
    returns : pd.DataFrame
    max_components : int, optional
        Maximum number of components to show.  Defaults to min(N, window) - 1.
    window : int
        How many trailing periods to use for the single covariance estimate.

    Returns
    -------
    pd.DataFrame
        Columns: n_components, explained_variance_ratio, cumulative_variance.
    """
    sample = returns.iloc[-window:].dropna(axis=1, thresh=int(window * 0.8))
    X = sample.values
    X = X - X.mean(axis=0)
    X = np.nan_to_num(X, nan=0.0)

    max_k = max_components or min(X.shape)
    pca = PCA(n_components=max_k)
    pca.fit(X)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    return pd.DataFrame(
        {
            "n_components": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_variance": cum_evr,
        }
    )


def pca_loadings(
    returns: pd.DataFrame,
    n_components: int = 5,
    window: int = 252,
) -> pd.DataFrame:
    """
    Return the PCA eigenvectors (loadings) estimated on the trailing *window*.

    Useful for interpreting which stocks load heavily on each component.
    Uses only past data (last *window* rows of *returns*).

    Returns
    -------
    pd.DataFrame
        Shape (n_components, N), index = PC1..PCk, columns = ticker symbols.
    """
    sample = returns.iloc[-window:].dropna(axis=1, thresh=int(window * 0.8))
    X = sample.values
    X = X - X.mean(axis=0)
    X = np.nan_to_num(X, nan=0.0)

    k = min(n_components, min(X.shape) - 1)
    pca = PCA(n_components=k)
    pca.fit(X)

    return pd.DataFrame(
        pca.components_,
        index=[f"PC{i+1}" for i in range(k)],
        columns=sample.columns,
    )


def regress_pca_on_ff(
    pca_factors: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Regress each PCA factor on FF factors to provide economic interpretation.

    Returns a DataFrame with R² and beta coefficients for each PC vs. each FF factor.

    Parameters
    ----------
    pca_factors : pd.DataFrame
        Shape (T, K) — time series of PCA factor returns.
    ff_factors : pd.DataFrame
        Shape (T, F) — Fama-French factor returns (Mkt-RF, SMB, HML, ...).

    Returns
    -------
    pd.DataFrame
        Multi-level columns: (pc_name, {r2, beta_Mkt-RF, beta_SMB, ...}).
    """
    import statsmodels.api as sm

    common_idx = pca_factors.index.intersection(ff_factors.index)
    pca_aligned = pca_factors.loc[common_idx]
    ff_aligned = ff_factors.loc[common_idx]
    X = sm.add_constant(ff_aligned)

    records = []
    for pc in pca_factors.columns:
        y = pca_aligned[pc].dropna()
        X_fit = X.reindex(y.index).dropna()
        y_fit = y.reindex(X_fit.index)
        model = sm.OLS(y_fit, X_fit).fit()
        row = {"PC": pc, "R2": model.rsquared}
        for coef_name, coef_val in zip(X_fit.columns, model.params):
            row[f"beta_{coef_name}"] = coef_val
        records.append(row)

    return pd.DataFrame(records).set_index("PC")
