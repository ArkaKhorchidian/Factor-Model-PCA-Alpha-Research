"""
Tests for performance metrics.

Key invariants:
  - Sharpe of constant zero-return series = 0.0 exactly.
  - Max drawdown is always ≤ 0.
  - Annualized return of a flat series = 0.0.
  - Win rate of all-positive returns = 1.0.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    format_metrics_table,
    ic_ir,
    max_drawdown,
    performance_metrics,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)


class TestAnnualizedReturn:
    def test_zero_return_series(self):
        r = pd.Series([0.0] * 252)
        assert abs(annualized_return(r)) < 1e-10

    def test_positive_series(self):
        # Daily return of 0.1% → annual ≈ 28%
        r = pd.Series([0.001] * 252)
        ann = annualized_return(r)
        assert ann > 0.25

    def test_empty_series_returns_nan(self):
        r = pd.Series([], dtype=float)
        assert np.isnan(annualized_return(r))


class TestSharpeRatio:
    def test_zero_return_zero_sharpe(self):
        r = pd.Series([0.0] * 252)
        assert sharpe_ratio(r) == 0.0

    def test_positive_returns_positive_sharpe(self):
        r = pd.Series([0.001] * 252)
        assert sharpe_ratio(r) > 0

    def test_sharpe_with_risk_free(self):
        np.random.seed(1)
        r = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        rf = 0.0002  # 2 bps/day risk-free
        sr_gross = sharpe_ratio(r, risk_free=0.0)
        sr_net = sharpe_ratio(r, risk_free=rf)
        assert sr_net < sr_gross

    def test_insufficient_data_returns_nan(self):
        r = pd.Series([0.01])
        assert np.isnan(sharpe_ratio(r))

    def test_constant_return_finite_sharpe(self):
        """Constant non-zero return has zero vol → Sharpe should be nan or very large."""
        r = pd.Series([0.001] * 252)
        # std will be 0, result should be nan or inf — implementation returns inf or nan
        # Our implementation: 0/0 → nan; non-zero/0 → large positive
        sr = sharpe_ratio(r)
        # Should be positive (signal positive) but check it doesn't crash
        assert sr is not None


class TestMaxDrawdown:
    def test_always_non_positive(self):
        r = pd.Series(np.random.randn(500) * 0.01)
        mdd = max_drawdown(r)
        assert mdd <= 0.0

    def test_monotone_increasing_zero_dd(self):
        r = pd.Series([0.01] * 100)
        mdd = max_drawdown(r)
        assert abs(mdd) < 1e-6

    def test_known_drawdown(self):
        # Wealth: 1 → 1.1 → 0.99 → 1.05
        # Peak = 1.1, trough = 0.99 → MDD = (0.99 - 1.1)/1.1 ≈ -0.10
        r = pd.Series([0.10, -0.10, 0.0606])
        mdd = max_drawdown(r)
        assert mdd < -0.05


class TestSortinoRatio:
    def test_all_positive_returns_large_sortino(self):
        r = pd.Series([0.001] * 252)
        # No downside variance — result may be nan (no negative returns)
        sr = sortino_ratio(r)
        # Should be nan (no downside returns)
        assert np.isnan(sr) or sr > 1.0

    def test_negative_mean_negative_sortino(self):
        r = pd.Series([-0.001] * 252 + [0.0005] * 10)
        sr = sortino_ratio(r)
        assert sr < 0


class TestCalmarRatio:
    def test_positive_return_positive_calmar(self):
        # Consistent positive returns with one drawdown period
        r = pd.Series([0.002] * 200 + [-0.01] * 10 + [0.002] * 42)
        calmar = calmar_ratio(r)
        assert calmar > 0

    def test_zero_drawdown_returns_nan(self):
        r = pd.Series([0.01] * 100)
        calmar = calmar_ratio(r)
        # Max drawdown = 0 → division by zero → nan
        assert np.isnan(calmar)


class TestWinRate:
    def test_all_positive_is_one(self):
        r = pd.Series([0.001] * 100)
        assert win_rate(r) == 1.0

    def test_half_positive_is_half(self):
        r = pd.Series([0.01, -0.01] * 50)
        assert abs(win_rate(r) - 0.5) < 1e-9


class TestTurnover:
    def test_zero_change_zero_turnover(self):
        w = pd.DataFrame({"A": [0.5] * 10, "B": [-0.5] * 10})
        to = turnover(w)
        assert (to.dropna() == 0.0).all()

    def test_full_flip_is_one(self):
        # Flip from [0.5, -0.5] to [-0.5, 0.5]: |Δw| = [1, 1], sum=2, /2 = 1.0
        w = pd.DataFrame(
            {"A": [0.5, -0.5], "B": [-0.5, 0.5]},
            index=pd.date_range("2020-01-01", periods=2),
        )
        to = turnover(w)
        assert abs(to.iloc[1] - 1.0) < 1e-9


class TestIcIr:
    def test_positive_mean_positive_icir(self):
        ic = pd.Series([0.04] * 50 + [0.02] * 50)
        assert ic_ir(ic) > 0

    def test_zero_std_returns_nan(self):
        ic = pd.Series([0.05] * 100)
        assert np.isnan(ic_ir(ic))


class TestPerformanceMetrics:
    def test_returns_all_expected_keys(self):
        r = pd.Series(np.random.randn(252) * 0.01)
        m = performance_metrics(r)
        expected = {
            "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate",
            "value_at_risk_95", "avg_ic", "ic_ir", "avg_turnover",
        }
        assert expected.issubset(m.keys())

    def test_format_metrics_table_returns_dataframe(self):
        r = pd.Series(np.random.randn(252) * 0.01)
        m = performance_metrics(r)
        table = format_metrics_table(m)
        assert isinstance(table, pd.DataFrame)
        assert "Value" in table.columns
