# Research Results

This page contains detailed research-result summaries and links to the source notebooks.

For the notebook inventory and published HTML report URLs, see [Notebook Catalog](notebooks.md).

## Realized Volatility Forecasting (HAR-RV-VIX vs RF)

We build a **21-day realized variance** forecasting model on ES futures (2010-2025) and compare:

- **Naive RV benchmark** - carry current 21D RV forward
- **HAR-RV** - classic daily / weekly / monthly RV lags
- **HAR-RV-VIX** - HAR-RV + VIX as a forward-looking volatility proxy
- **Random Forest (RF)** - non-linear benchmark on the same feature set

Before model selection, we run a **feature-importance & stability analysis** (SFI, Lasso, RF, permutation importance) to keep only **parsimonious, economically sensible** predictors for the linear and RF models.

![Feature Importance Linear](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/fi_linear.png)

### OOS Performance (2021-2025, monthly walk-forward, 3Y rolling window)

All metrics are computed on **log 21D RV**, using an expanding walk-forward with **3-year rolling re-fit** and a **21-day purge**.

| model      |   R^2   |   MSE    |  QLIKE   | Var_res | R^2_oos |
|:-----------|:-------:|:--------:|:--------:|:-------:|:-------:|
| Naive_RV   | 0.0943  | 0.5078   | 0.2791   | 0.5080  | 0.0000  |
| HAR-RV     | 0.2920  | 0.3970   | 0.2086   | 0.3920  | 0.2182  |
| HAR-RV-VIX | 0.3676  | 0.3546   | 0.1788   | 0.3549  | 0.3017  |

![RV OOS Performance](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/rv_oos_perf.png)

### Takeaways

- **HAR-RV-VIX** is the **final candidate model**: it clearly beats both **Naive RV** and **HAR-RV** in OOS R^2, MSE and QLIKE, and delivers about **30% R^2_oos** vs the naive benchmark.
- The **Random Forest** does **not** improve on HAR-RV-VIX in the validation period and is therefore **not carried forward** to the final walk-forward evaluation.
- All modelling choices (features, models, hyper-parameters) were fixed on **2010-2020**; the **2021-2025** walk-forward backtest is run once to avoid backtest-tuning bias.

Source notebook: [rv_forecasting](https://github.com/anthonymakarewicz/volatility-trading/blob/main/notebooks/rv_forecasting/notebook.ipynb)

## Implied Volatility Surface Modelling (Parametric vs Non-Parametric)

![IV Surface](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/iv_surface_grid.png)

Source notebook: [iv_surface_modelling](https://github.com/anthonymakarewicz/volatility-trading/blob/main/notebooks/iv_surface_modelling/notebook.ipynb)

## Skew Volatility Trading (30 DTE / 25 Delta)

Trade the 30-day to expiry, 25 delta SPX put-call skew via a delta-hedged risk reversal.

### Entry/Exit Logic

- **Short RR** when skew z-score >= 1.5 (too steep)
- **Long RR** when skew z-score <= -1.5 (too flat)
- **Exit** when |z-score| <= 0.5

![Abs vs Norm Skew](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/abs_vs_norm_skew.png)
![Skew Z-score](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/z_score_signal_vix_filter.png)

### Signal Filters

- **VIX filter:** block entries if VIX > 30
- **IV percentile:** trade only when ATM IV is within its 20-80 historical percentile
- **Skew percentile:** trade only when skew is below its 30th (for longs) or above its 70th (for shorts) percentile

### Backtest Overview

Walk-forward backtest on daily SPX options (2016-2023), starting with $100,000 capital.

### Configuration Snapshot

- **Entry rule:** 50-day z-score mean reversion on normalized skew (|z| >= 1.5 triggers a risk reversal; exit when |z| < 0.5)
- **Execution costs:** sell at bid, buy at ask + $0.01 slippage per leg, $1 commission per option leg
- **Risk controls:** delta hedging via ES futures, dynamic sizing by signal strength, risk floor, stop-loss/take-profit, holding-period cap

Source notebook: [skew_trading](https://github.com/anthonymakarewicz/volatility-trading/blob/main/notebooks/skew_trading/notebook.ipynb)

### Performance Results

![Backtest](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/backtest_baseline_realistic.png)
![PnL Decomposition](https://raw.githubusercontent.com/anthonymakarewicz/volatility-trading/main/plots/pnl_decomp_basline_realistic.png)

| Metric                    | Value          |
|:--------------------------|:--------------:|
| **Sharpe Ratio**          | 0.61           |
| **CAGR**                  | 10.73%         |
| **Average Drawdown**      | -1.62%         |
| **Max Drawdown**          | -14.61%        |
| **Max Drawdown Duration** | 287 days       |
| **Historical VaR (99%)**  | -0.87%         |
| **Historical CVaR (99%)** | -2.64%         |
| **Total P&L**             | $68,754.99     |
| **Profit Factor**         | 2.66           |
| **Trade Frequency**       | 15.2 trades/yr |
| **Total Trades**          | 78             |
| **Win Rate**              | 62.82%         |
| **Avg Win P&L**           | $2,271.04      |
| **Avg Loss P&L**          | -$1,444.52     |
