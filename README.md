# Factor Model & PCA Alpha Research

A quantitative equity research implementation covering the full workflow from
factor construction to alpha extraction, portfolio backtesting, and performance
attribution. Built to professional research standards with explicit treatment of
survivorship bias, transaction costs, and honest Sharpe reporting.

---

## Research Questions

| Notebook | Question |
|----------|----------|
| [01 — FF Replication](notebooks/01_ff_factor_replication.ipynb) | Can we replicate Fama-French 3/5 factors and verify the data pipeline? |
| [02 — PCA Extraction](notebooks/02_pca_factor_extraction.ipynb) | What statistical factors explain cross-sectional return covariance? |
| [03 — Momentum Signal](notebooks/03_momentum_signal.ipynb) | Does 12-1 month momentum have predictive power? At what horizons? |
| [04 — Composite Alpha](notebooks/04_composite_alpha.ipynb) | Does combining signals improve IC vs. the best single signal? |
| [05 — Portfolio Backtest](notebooks/05_portfolio_backtest.ipynb) | What is the gross and net Sharpe of the long/short strategy? |
| [06 — Attribution](notebooks/06_performance_attribution.ipynb) | How much of the return is true alpha vs. factor beta exposure? |

---

## Methodology

### Universe & Data Sources

**Price data**: Yahoo Finance via `yfinance` (daily adjusted close).

**Factor data**: Kenneth French's Data Library via `pandas-datareader`.

**Universe**: ~100 large-cap US equities (S&P 100 approximate proxy).

> **Survivorship Bias Disclosure**: The proxy universe uses *current* S&P 100
> constituents as of 2024. All firms in this list survived to the present, which
> biases historical backtests upward. The true magnitude of the bias is unknown
> without point-in-time constituent data (available via CRSP/Compustat on WRDS).
> All reported Sharpe ratios should be discounted by approximately 20–40% to
> account for this bias, per typical findings in the literature.
>
> Preferred mitigation: replace with point-in-time CRSP constituent lists.
> Minimum viable mitigation implemented here: broad universe + explicit disclosure.

### Factor Construction

**Momentum (12-1)**: Cumulative return from t-252 to t-21 trading days
(skip last month to avoid short-term reversal contamination).
Cross-sectionally ranked, winsorized at 1st/99th percentile, z-scored.

Reference: Jegadeesh & Titman (1993), Eq. (1).

**Short-term Reversal**: Negative of trailing 21-day return.
Captures mean-reversion at 1-month horizon.

**PCA Factors**: Rolling PCA on the trailing 252-day return covariance matrix.
Fit strictly in-sample (no look-ahead). Top K components retaining ~80% of variance.

> **PCA Caveat**: Statistical factors are NOT tradeable portfolios. They describe
> the low-dimensional structure of return covariance. Fitting PCA on the full
> sample and backtesting in-sample would dramatically inflate results (look-ahead bias).

### Signal Combination

IC-weighted composite:

```
weight_i = IC_i / Σ|IC_j|
composite = Σ weight_i × signal_i
```

Rolling 252-day IC estimation window prevents look-ahead in the combination step.

Signals are orthogonalized against FF5 factor exposures via cross-sectional OLS
residualization to isolate pure alpha from systematic beta.

Reference: Grinold & Kahn (1999), Ch. 14.

### Portfolio Construction & Transaction Costs

- **Portfolio**: Dollar-neutral long/short — long top quintile, short bottom quintile.
- **Rebalancing**: Monthly (approximately 21 trading days).
- **Transaction cost model**: 5 bps one-way (conservative for liquid large-caps).
- **Net return**: `gross - 2 × 5bps × turnover` per period.

The factor of 2 accounts for both legs of a round trip.
Both gross and net Sharpe are reported. If the spread collapses on a net basis, that is stated explicitly.

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `annualized_return` | Geometric annualized return |
| `annualized_volatility` | Std × √252 |
| `sharpe_ratio` | (E[R-Rf]) / σ × √252 |
| `sortino_ratio` | Return / downside vol |
| `max_drawdown` | Peak-to-trough |
| `calmar_ratio` | Ann. return / \|MDD\| |
| `avg_ic` | Mean cross-sectional rank IC |
| `ic_ir` | mean(IC) / std(IC) — signal consistency |
| `avg_turnover` | Fraction of portfolio traded per period |

> IC and IC IR are the metrics quant researchers care about most.
> An IC IR > 0.5 with mean IC ≈ 0.03–0.05 is a credible, non-overfit signal.

---

## Results

*See individual notebooks for full results. Summary from notebook 05:*

| Metric | Gross | Net |
|--------|-------|-----|
| Sharpe Ratio | — | — |
| Ann. Return | — | — |
| Max Drawdown | — | — |
| Avg Turnover | — | — |
| Avg IC | — | — |
| IC IR | — | — |

*Run notebooks to populate. Sample tearsheet: [results/tearsheet_sample.png](results/tearsheet_sample.png).*

---

## Reproducing Results

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Run notebooks (in order)
jupyter notebook notebooks/
```

Notebooks cache downloaded data to `data/cache/` (gitignored). First run
requires an internet connection. Subsequent runs use local cache.

---

## Project Structure

```
factor-model-alpha/
├── src/
│   ├── data/
│   │   ├── loader.py           # Price/return data fetching with disk cache
│   │   ├── ff_factors.py       # Fama-French 3/5-factor data
│   │   └── universe.py         # Stock universe, survivorship bias notes
│   ├── factors/
│   │   ├── classic.py          # SMB, HML, momentum factor construction
│   │   ├── technical.py        # Momentum, reversal, vol, liquidity signals
│   │   ├── pca_factors.py      # Rolling PCA factor extraction
│   │   └── combine.py          # IC computation, IC-weighted combination
│   ├── backtest/
│   │   ├── engine.py           # Backtest loop (gross + net P&L)
│   │   ├── portfolio.py        # Quintile weights, MV optimization (CVXPY)
│   │   ├── transaction_costs.py# Flat and proportional cost models
│   │   └── metrics.py          # Full suite: Sharpe, IC, IR, MDD, Calmar
│   ├── analysis/
│   │   ├── factor_attribution.py  # Barra-style return decomposition
│   │   ├── ic_analysis.py         # IC decay, sector IC, autocorrelation
│   │   └── tearsheet.py           # Multi-panel performance tearsheet
│   └── utils.py                # Winsorization, z-score, neutralization
├── notebooks/                  # One notebook per research question
├── tests/                      # pytest suite
├── data/cache/                 # Local data cache (gitignored)
└── results/                    # Tearsheet outputs
```

---

## Limitations & Future Work

1. **Survivorship bias**: Current S&P 100 proxy universe. Point-in-time constituents
   (CRSP via WRDS) would eliminate this and likely reduce reported Sharpe by 20–40%.

2. **Short selling constraints**: The model assumes frictionless shorting. In practice,
   borrow costs for some names can be 50–500 bps annually, which would further
   erode net performance for the short leg.

3. **Market impact**: The flat 5 bps cost model does not capture nonlinear price
   impact for larger position sizes. The Almgren-Chriss model would be more realistic
   for sizing above ~5% ADV.

4. **Parameter choices**: Formation window (12 months), skip window (1 month),
   quintile cutoffs, and rebalancing frequency were chosen based on the original
   Jegadeesh & Titman (1993) specification — NOT optimized to maximize Sharpe on
   this specific universe/sample. Post-hoc optimization would invalidate all
   statistical inference.

5. **Fundamental data**: True FF5 replication (RMW, CMA) requires accounting data
   (book value, operating profitability, investment) from Compustat. We use proxies
   where unavailable.

6. **Regime dependence**: Momentum strategies famously crash during sharp reversals
   (e.g., 2009 momentum crash). Regime-aware position sizing (not implemented) would
   improve robustness.

**Future work**: Hidden Markov Model regime detection, GARCH/HAR volatility forecasting
for dynamic risk targeting, and alternative data integration.

---

## References

1. Fama, E.F. & French, K.R. (1993). "Common risk factors in the returns on stocks
   and bonds." *Journal of Financial Economics*, 33(1), 3–56.

2. Fama, E.F. & French, K.R. (2015). "A five-factor asset pricing model."
   *Journal of Financial Economics*, 116(1), 1–22.

3. Jegadeesh, N. & Titman, S. (1993). "Returns to buying winners and selling losers:
   Implications for stock market efficiency." *Journal of Finance*, 48(1), 65–91.

4. Grinold, R. & Kahn, R. (1999). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
   *(IC/IR framework, Fundamental Law of Active Management)*

5. Connor, G. & Korajczyk, R. (1988). "Risk and return in an equilibrium APT:
   Application of a new test methodology." *Journal of Financial Economics*, 21(2), 255–289.

6. Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). "The Cross-Section of Volatility
   and Expected Returns." *Journal of Finance*, 61(1), 259–299.

7. Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series
   Effects." *Journal of Financial Markets*, 5(1), 31–56.
