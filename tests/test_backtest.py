"""
Tests for backtest engine and portfolio construction.

Checks:
  - Dollar neutrality maintained each period (long $ ≈ short $).
  - Backtest engine produces non-trivial returns.
  - Quintile weights sum correctly.
  - Transaction cost drag is non-negative.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import cumulative_returns, run_backtest
from src.backtest.portfolio import (
    build_portfolio_weights,
    quintile_weights,
    verify_dollar_neutral,
)
from src.backtest.transaction_costs import apply_flat_costs, cost_drag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_data(T: int = 500, N: int = 30, seed: int = 42):
    np.random.seed(seed)
    dates = pd.date_range("2018-01-02", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    returns = pd.DataFrame(
        np.random.randn(T, N) * 0.01,
        index=dates,
        columns=tickers,
    )
    signal = pd.DataFrame(
        np.random.randn(T, N),
        index=dates,
        columns=tickers,
    )
    return returns, signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


class TestQuintileWeights:
    def test_sums_near_zero(self):
        s = pd.Series(np.linspace(-1, 1, 50))
        w = quintile_weights(s)
        assert abs(w.sum()) < 1e-9

    def test_long_leg_positive(self):
        s = pd.Series(np.linspace(-1, 1, 50))
        w = quintile_weights(s)
        assert (w[w > 0]).sum() > 0

    def test_short_leg_negative(self):
        s = pd.Series(np.linspace(-1, 1, 50))
        w = quintile_weights(s)
        assert (w[w < 0]).sum() < 0

    def test_long_leg_sums_to_one(self):
        s = pd.Series(np.linspace(-1, 1, 50))
        w = quintile_weights(s)
        assert abs(w[w > 0].sum() - 1.0) < 1e-9

    def test_short_leg_sums_to_neg_one(self):
        s = pd.Series(np.linspace(-1, 1, 50))
        w = quintile_weights(s)
        assert abs(w[w < 0].sum() + 1.0) < 1e-9

    def test_insufficient_data_returns_zeros(self):
        s = pd.Series([1.0, 2.0])  # less than 10 stocks
        w = quintile_weights(s)
        assert (w == 0).all()


class TestBuildPortfolioWeights:
    def test_dollar_neutral_each_period(self):
        returns, signal = _synthetic_data(T=300, N=30)
        weights = build_portfolio_weights(signal, rebalance_freq="ME")
        neutral = verify_dollar_neutral(weights, tol=0.1)
        # Allow a small number of violations (e.g., at start-up)
        assert neutral.mean() > 0.9, f"Dollar neutrality violated: {neutral.mean():.2%}"

    def test_weights_shape(self):
        returns, signal = _synthetic_data(T=300, N=30)
        weights = build_portfolio_weights(signal, rebalance_freq="ME")
        assert weights.shape == signal.shape

    def test_rebalance_dates_match_frequency(self):
        returns, signal = _synthetic_data(T=300, N=30)
        weights = build_portfolio_weights(signal, rebalance_freq="ME")
        # Weights should change only at month boundaries
        diffs = weights.diff().abs().sum(axis=1)
        n_changes = (diffs > 1e-9).sum()
        n_months = len(signal.resample("ME").last())
        # Should have approximately n_months rebalances (+1 for initial)
        assert n_changes <= n_months + 2


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_returns_correct_keys(self):
        returns, signal = _synthetic_data()
        weights = build_portfolio_weights(signal)
        result = run_backtest(weights, returns)
        assert "gross_returns" in result
        assert "net_returns" in result
        assert "turnover" in result
        assert "gross_metrics" in result
        assert "net_metrics" in result

    def test_net_sharpe_leq_gross_sharpe(self):
        """Net Sharpe should be ≤ gross Sharpe after transaction costs."""
        returns, signal = _synthetic_data()
        weights = build_portfolio_weights(signal)
        result = run_backtest(weights, returns, one_way_cost_bps=5.0)
        gross_sr = result["gross_metrics"]["sharpe_ratio"]
        net_sr = result["net_metrics"]["sharpe_ratio"]
        assert net_sr <= gross_sr + 1e-6

    def test_cumulative_returns_starts_at_100(self):
        r = pd.Series([0.01, -0.005, 0.02, -0.01])
        cum = cumulative_returns(r, start_value=100.0)
        assert abs(cum.iloc[0] - 101.0) < 1e-6  # first period applied

    def test_gross_return_length(self):
        returns, signal = _synthetic_data(T=300)
        weights = build_portfolio_weights(signal)
        result = run_backtest(weights, returns)
        assert len(result["gross_returns"]) == len(returns)


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_positive_turnover_reduces_return(self):
        gross = pd.Series([0.01] * 100)
        to = pd.Series([0.5] * 100)  # 50% turnover
        net = apply_flat_costs(gross, to, one_way_cost_bps=5.0)
        expected_cost = 2 * 5e-4 * 0.5
        expected_net = 0.01 - expected_cost
        assert abs(net.mean() - expected_net) < 1e-10

    def test_zero_turnover_no_cost(self):
        gross = pd.Series([0.01] * 100)
        to = pd.Series([0.0] * 100)
        net = apply_flat_costs(gross, to, one_way_cost_bps=5.0)
        pd.testing.assert_series_equal(gross, net.rename(gross.name))

    def test_cost_drag_non_negative(self):
        drag = cost_drag(gross_sharpe=1.2, net_sharpe=0.9)
        assert drag["sharpe_drag"] >= 0
        assert drag["drag_pct"] >= 0

    def test_cost_drag_zero_if_equal(self):
        drag = cost_drag(gross_sharpe=1.0, net_sharpe=1.0)
        assert drag["sharpe_drag"] == 0.0
