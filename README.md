# **Volatility Trading on Index Equity Options**

This projects presents several daily volatility trading strategies on SPX Index Equity Options.
Each strategy is thoroughly backtested and appropriate risk management constraints are considered.


## **Realized Volatility Forecasting (HAR-RV-VIX vs RF)**

We build a **21-day realized variance** forecasting model on ES futures (2010–2025) and compare:

- **Naive RV benchmark** – carry current 21D RV forward  
- **HAR-RV** – classic daily / weekly / monthly RV lags  
- **HAR-RV-VIX** – HAR-RV + VIX as a forward-looking volatility proxy  
- **Random Forest (RF)** – non-linear benchmark on the same feature set

Before model selection, we run a **feature-importance & stability analysis** (SFI, Lasso, RF, permutation importance) to keep only **parsimonious, economically sensible** predictors for the linear and RF models.

![alt text](image-3.png)

---

### OOS Performance (2021–2025, monthly walk-forward, 3Y rolling window)

All metrics are computed on **log 21D RV**, using an expanding walk-forward with **3-year rolling re-fit** and a **21-day purge**.

| model      |   $R²$    |   MSE    |  QLIKE   | Var_res | $R²_{oos}$|
|:----------|:-------:|:--------:|:--------:|:-------:|:------:|
| Naive_RV  | 0.0943  | 0.5078   | 0.2791   | 0.5080  | 0.0000 |
| HAR-RV    | 0.2920  | 0.3970   | 0.2086   | 0.3920  | 0.2182 |
| HAR-RV-VIX| 0.3676  | 0.3546   | 0.1788   | 0.3549  | 0.3017 |


![alt text](image-2.png)

---

###  Takeaways

- **HAR-RV-VIX** is the **final candidate model**: it clearly beats both **Naive RV** and **HAR-RV** in OOS $R²$, MSE and QLIKE, and delivers a **~30% $R²_{oos}$** vs the naive benchmark.  
- The **Random Forest** does **not** improve on HAR-RV-VIX in the validation period and is therefore **not carried forward** to the final walk-forward evaluation.  
- All modelling choices (features, models, hyper-parameters) were fixed on **2010–2020**; the **2021–2025** walk-forward backtest is run **once** to avoid backtest-tuning bias.

👉 Full notebook: `notebooks/rv_forecasting.ipynb`


## **Implied Volatility Surface Modelling (Parametric vs Non-Parametric)**

![Iv surface](plots/iv_surface_grid.png)

👉 Full notebook: `notebooks/iv_surface_modelling.ipynb`


## **Skew Volatility Trading (30 DTE / 25 Δ)**

We run a walk-forward backtest on daily SPX options (2016 – 2023), starting with \$100 000 of capital.

### Configuration

- **Entry Rule**  
50-day z-score mean-reversion on normalized skew (|z| ≥ 1.5 triggers a risk reversal; exit when |z| falls below 0.5)

- **Signal Filters**  
  - **VIX Filter:** Skip trades when VIX > 30  
  - **Skew Percentile:** Only go long when normalized skew falls below its 30th percentile, and short when it rises above its 70th percentile   

- **Execution Costs**  
  - **Bid/Ask & Slippage:** Fill sells at the bid, buys at the ask + \$0.01 slippage per leg  
  - **Commissions:** \$1 per option leg  

- **Risk Controls**
  - **Delta Hedging:** E-mini S&P 500 futures (ES=F) used to neutralize net Δ (lot size = 50)
  - **Position Sizing:** Dynamically scale trade size by signal strength (base 1% of equity at entry-threshold, +0.5% per additional 0.5σ) and cap it at 2%
  - **Risk Floor:** Enforce a minimum \$750 worst-case risk per contract to prevent oversized position sizing when Greek-based risk is very low    
  - **Stop-Loss & Take-Profit:** SL at 100% of notional, TP at 70% of notional  
  - **Holding Period Cap:** 3 business days (skip negative theta trades on Fridays if 2-day θ decay > 200)

👉 Full notebook: `notebooks/skew_trading.ipynb`

---

### Performance Results

![Backtest](plots/backtest_baseline_realistic.png)
![alt text](plots/pnl_decomp_basline_realistic.png)

---

### 🔍 **Overall Performance Metrics**

| Metric                    | Value          |
|:--------------------------|:--------------:|
| **Sharpe Ratio**          | 0.61           |
| **CAGR**                  | 10.73%         |
| **Average Drawdown**      | –1.62%         |
| **Max Drawdown**          | –14.61%        |
| **Max Drawdown Duration** | 287 days       |
| **Historical VaR (99%)**  | –0.87%         |
| **Historical CVaR (99%)** | –2.64%         |
| **Total P&L**             | \$68,754.99    |
| **Profit Factor**         | 2.66           |
| **Trade Frequency**       | 15.2 trades/yr |
| **Total Trades**          | 78             |
| **Win Rate**              | 62.82%         |
| **Avg Win P&L**           | \$2,271.04     |
| **Avg Loss P&L**          | –\$1,444.52    |