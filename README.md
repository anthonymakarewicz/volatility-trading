# **Volatility Trading on Index Equity Options**

This projects presents several daily volatility trading strategies on SPX Index Equity Options.
Each strategy is thoroughly backtested and appropriate risk management constraints are considered.

## **Skew Trading**

We use the **30-DTE 25-Delta Skew** to capture the behaviour of the skew. To express our view 
on the skew being abnormally steep or flat, we use a **Risk Reversal structure**.
![Risk Reversal](plots/risk_reversal.png)

---

### **Trading signals using the Z-score & VIX Sentiment filter for trade quality**
To generate trading signals, we use a **rolling Z-score** of this skew to capture large deviations.
In addition, are filtered by the **VIX** to avoid trading durign panic regimes.
![Skew Z-score](plots/z_score_signal_vix_filter.png)

---

### **Backtest using Walk-Forward cross validation, risk management, and realistic backtest constraints**
This is the result of a backtest using a **rolling Skew percentile** with thresholds 20% and 80% used to validate each trade.
![Backtest](plots/backtest_greeks.png)

### 🔍 **Overall Performance Metrics**

| Metric                   | Value        |
|:-------------------------|:------------:|
| **Sharpe Ratio**         | 1.12         |
| **CAGR**                 | 15.06%       |
| **Max Drawdown**         | -3.83%       |
| **Max Drawdown Duration**| 226 days     |
| **Total P&L**            | \$59,855     |
| **Profit Factor**        | 3.49         |
| **Trade Frequency**      | 9.8 trades/yr|
| **Total Trades**         | 34           |
| **Win Rate**             | 61.76%       |
| **Avg Win P&L**          | \$3,993.76   |
| **Avg Loss P&L**         | -\$1,847.23  |

---

### 📊 **Performance by Contract Size**

| Contracts | Win Rate | # Trades | Total Win P&L | Total Loss P&L | Total P&L |
|:---------:|:--------:|:--------:|:-------------:|:--------------:|:---------:|
| 1         | 62%      | 29       | \$54,739      | -\$19,785      | \$34,954  |
| 2         | 75%      | 4        | \$29,130      | -\$2,232       | \$26,898  |
| 3         | 0%       | 1        | \$0           | -\$1,997       | -\$1,997  |
