# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: volatility_trading
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Forecasting Realized Volatility**
#
# In contrast to *implied volatility* (IV), which reflects the market’s expectation of future uncertainty, *realized volatility* (RV) corresponds to the volatility that actually materializes over a given horizon. From a modeling perspective, RV exhibits strong persistence and clustering, making it more forecastable than IV or returns themselves.
#
# ## Why Forecast Realized Volatility?
#
# Forecasting RV is useful for both trading and risk management:
#
# - **Volatility trading / IV–RV strategies** – RV forecasts help detect mispricing between implied and realized volatility and quantify the volatility risk premium (VRP).  
# - **Position sizing** – scale exposure up in calm regimes, down in turbulent ones.  
# - **Risk management** – set realistic stop-loss / take-profit levels and scenario ranges for portfolio moves.
#
#
# ## Research Questions
#
# In this notebook we focus on 21-day realized variance and ask:
#
# 1. **Can we extend HAR-RV in a parsimonious and interpretable way?**  
#    In particular, does a `HAR-RV-VIX` specification provide a robust improvement over standard HAR-RV?
#
# 2. **Do non-linear models add value?**  
#    Can a Random Forest capture useful **non-linearities and interactions** between predictors that a linear HAR-RV-X cannot?
#
# 3. **Can we beat simple benchmarks?**  
#    Is there a linear HAR-RV-X, a non-linear RF, or an ensemble of both that consistently outperforms:
#    - a **Naive RV** benchmark (carry the current 21D RV forward), and  
#    - a standard **HAR-RV** benchmark?
#
#
# ## Summary
#
# The notebook is structured as follows:
#
# 1. [Read & Prepare Data](#read--prepare-data)
# 2. [Volatility Estimators](#volatility-estimators)
#    - [2.1 Historical (close-to-close)](#21-historical-close-to-close)
#    - [2.2 Range-based (OHLC)](#22-range-based-ohlc)
#    - [2.3 High-frequency realized variance](#23-high-frequency-estimators-realized-variance)
# 3. [Stylized Facts of Realized Volatility](#stylized-facts-of-daily-volatility)
# 4. [Problem Formulation](#problem-formulation)
#    - [4.1 Target: 21-day realized variance](#41-target-variable-21-day-realized-variance)
#    - [4.2 Predictor families](#42-predictor-families)
# 5. [Feature Engineering: Rolling and regime-switching features](#real-vol)
# 6. [Data-Preprocessing: Data transformation, scaling and redundant features removal](#real-vol)
# 7. [Modelling Framework](#methodology)
#    - [7.1 Cross-validation (Purged K-Fold)](#71-cross-validation-scheme-purged-k-fold)
#    - [7.2 Model specifications](#72-model-specifications)
#    - [7.3 Evaluation metrics](#73-evaluation-metrics)
# 8. [Feature Importance & Stability](#8-feature-importance-and-stability-analysis)
#    - [8.1 Single-feature importance (SFI)](#81-single-feature-importance-sfi)
#    - [8.2 In-model stability (Lasso / RF)](#82-in-model-stability-analysis-with-substitution-effects)
#    - [8.3 OOS permutation importance](#83-out-of-sample-permutation-importance-mean-decrease-accuracy)
# 9. [Model Selection: Linear, Random Forest and Ensembles](#9-model-selection)
# 10. [Out-of-Sample Walk-Forward Evaluation](#10-out-of-sample-walk-forward-evaluation)
# 11. [Conclusion](#11-conclusion)

# %%
# %load_ext autoreload
# %autoreload 2

import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from config.paths import DATA_INTER, DATA_PROC
from matplotlib import gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

import volatility_trading.rv_forecasting.features as rvfeat
import volatility_trading.rv_forecasting.plotting as ph
import volatility_trading.rv_forecasting.vol_estimators as rvvol
from volatility_trading.iv_surface.ssvi_model import SSVI
from volatility_trading.rv_forecasting.data_loading import load_intraday_prices
from volatility_trading.rv_forecasting.modelling import (
    DataProcessor,
    PurgedKFold,
    WalkForwardOOS,
    compute_metrics,
    compute_subperiod_metrics,
    eval_ensembles,
    eval_model_cv,
    in_sample_stability,
    oos_perm_importance,
    single_feature_importance,
)

np.random.seed(42)
random.seed(42)

# %matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')

pd.options.mode.chained_assignment = None 

# %% [markdown]
# # **1. Read & Prepare Data**
#
# For this analysis we are going to consider daily S&P500 OHLC data from Yahoo finance for a research period from `2010` to `2020`. The last years of data from `2021` to late `2025` will be imported at the end for the backtest so that we don't introduce a look ahead bais in our results.

# %%
start = "2010-01-01"
end = "2020-12-31"

spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
spx.columns = spx.columns.droplevel("Ticker")
spx.columns.name = None
spx

# %%
spx["returns"] = np.log(spx["Close"] / spx["Close"].shift(1))
spx = spx.dropna()
spx

# %%
spx["returns"].plot(figsize=(12, 6))
plt.title("Log Returns")

# %% [markdown]
# # **2. Volatility Estimators**
#
# Volatility is a *latent* quantity — it cannot be observed directly. To evaluate forecasts and build predictors, we need to construct **proxies** from price data. Different estimators trade off bias vs efficiency depending on the sampling frequency and data available.

# %% [markdown]
# ## 2.1 Historical / Close-to-Close
#
# The simplest approach uses only closing price data and is often called the **close-to-close estimator**.   It computes realized volatility from consecutive daily returns. This estimator serves as a useful **benchmark**, but it suffers from several drawbacks: it ignores intraday variation, overnight jumps, and the information contained in opening, high, and low prices, which often makes it downward biased.
#
# Formally, over a horizon $H$:
#
# $$
# RV_{t,H} = \sqrt{\tfrac{252}{H} \sum_{j=1}^H r_{t+j}^2}, 
# \quad r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
# $$

# %%
spx["rv_close"] = rvvol.rv_close_to_close(spx["returns"], h=21)
spx["rv_close"].plot(figsize=(12, 6))
plt.show()

# %% [markdown]
# ## 2.2 Range-Based Proxies (OHLC)
#
# Instead of using only daily closing prices, we can exploit the full **Open–High–Low–Close (OHLC)** information to build more efficient volatility estimators.
# Over time, several **range-based measures** have been proposed, each improving on the simple close-to-close estimator in different ways.
#
# These estimators:
# - typically **reduce the variance** of the volatility estimate,
# - better capture **intraday price dynamics**,
# - but rely on specific modelling assumptions (e.g. no drift, no jumps) and each has its own limitations.
#
# In this section we review a few standard OHLC-based proxies and compare them to the close-to-close benchmark.

# %% [markdown]
# ### 2.2.1 Parkinson Estimator
#
# The **Parkinson (1980) estimator** uses the daily high–low range to capture intraday price variability.  
# By relying on the full range rather than just closing prices, it provides a more efficient estimate of volatility under the assumption of a **driftless geometric Brownian motion**.  
#
# However, it ignores opening and closing prices and is highly sensitive to jumps or bid–ask bounce at the extremes of the trading day.
#
# Formally, over a horizon $H$:
#
# $$
# RV^{\text{Parkinson}}_{t,H} 
# = \sqrt{ \frac{252}{4H \ln(2)} \sum_{j=1}^H \left[ \ln\!\left(\tfrac{H_{t+j}}{L_{t+j}}\right) \right]^2 }
# $$

# %%
spx["rv_parkinson"] = rvvol.rv_parkinson(spx["High"], spx["Low"], h=21)
spx["rv_parkinson"].plot(figsize=(12, 6))
plt.show()

# %% [markdown]
# ### 2.2.2 Garman–Klass Estimator
#
# The **Garman–Klass (1980) estimator** improves upon the Parkinson measure by incorporating not only the daily high and low, but also the open and close prices.  
# This allows it to capture more information about intraday price variation and reduce estimation variance under the assumption of a driftless geometric Brownian motion.  
#
# However, like the Parkinson estimator, it can be biased in the presence of significant drift or opening jumps, since it assumes zero drift and continuous trading.
#
# Formally, over a horizon $H$:
#
# $$
# RV^{\text{GK}}_{t,H} 
# = \sqrt{ \frac{252}{H} \sum_{j=1}^H 
# \left[ \tfrac{1}{2} \left( \ln\!\left(\tfrac{H_{t+j}}{L_{t+j}}\right) \right)^2 
# - (2\ln(2) - 1) \left( \ln\!\left(\tfrac{C_{t+j}}{O_{t+j}}\right) \right)^2 
# \right] }
# $$

# %%
spx["rv_gk"] = rvvol.rv_garman_klass(spx["Open"], spx["High"], spx["Low"], spx["Close"], h=21)
spx["rv_gk"].plot(figsize=(12, 6))
plt.show()

# %% [markdown]
# ### 2.2.3 Rogers–Satchell Estimator
#
# The Rogers–Satchell (1991) estimator was introduced to fix a key limitation of the Parkinson and Garman–Klass estimators: their assumption of **zero drift**. Unlike those, Rogers–Satchell is **drift-robust**, making it more suitable for assets that can trend over time.
#
# It still uses all four **OHLC** prices, but in a different functional form that explicitly allows for **nonzero expected returns**. However, like other OHLC-based estimators, it remains **sensitive to microstructure noise** and **large jumps** at the open or close.
#
# Formally, over a horizon $H$:
#
# $$
# RV^{\text{RS}}_{t,H} 
# = \sqrt{ \frac{252}{H} \sum_{j=1}^H 
# \left[ 
# \ln\!\left(\tfrac{H_{t+j}}{C_{t+j}}\right) \cdot \ln\!\left(\tfrac{H_{t+j}}{O_{t+j}}\right) 
# + \ln\!\left(\tfrac{L_{t+j}}{C_{t+j}}\right) \cdot \ln\!\left(\tfrac{L_{t+j}}{O_{t+j}}\right) 
# \right] }
# $$

# %%
spx["rv_rs"] = rvvol.rv_rogers_satchell(spx["Open"], spx["High"], spx["Low"], spx["Close"], h=21)
spx["rv_rs"].plot(figsize=(12, 6))
plt.show()

# %% [markdown]
# ### 2.2.4 Yang–Zhang Estimator
#
# The **Yang–Zhang (2000) estimator** combines three pieces of information: overnight variance, open-to-close variance, and the Rogers–Satchell range component.  
# It is **unbiased in the presence of drift**, less sensitive to opening jumps, and has lower estimation variance than Parkinson, Garman–Klass, or Rogers–Satchell taken alone.
#
# Formally, the daily Yang–Zhang variance is:
#
# $$
# \sigma^2_{YZ} = \sigma^2_O + k \, \sigma^2_C + (1-k)\, \sigma^2_{RS}
# $$
#
# where  
# - $\sigma^2_O = \big( \ln(O_t / C_{t-1}) \big)^2$ is the overnight variance,  
# - $\sigma^2_C = \big( \ln(C_t / O_t) \big)^2$ is the open-to-close variance,  
# - $\sigma^2_{RS}$ is the Rogers–Satchell variance component,  
# - $k \approx 0.34$ is a weight chosen to minimise bias and variance.  
#
# Aggregating over a horizon $H$:
#
# $$
# RV^{\text{YZ}}_{t,H} = \sqrt{ \frac{252}{H} \sum_{j=1}^H \sigma^2_{YZ,\,t+j} }.
# $$

# %%
spx["rv_yz"] = rvvol.rv_yang_zhang(spx["Open"], spx["High"], spx["Low"], spx["Close"], h=21)
spx["rv_yz"].plot(figsize=(12, 6))
plt.show()

# %% [markdown]
# ###  Which Estimator to choose ?

# %%
spx["overnight_volatility"] = np.log(spx["Open"] / spx["Close"].shift(1)).rolling(21).std()
spx["intraday_volatility"] = np.log(spx["Close"] / spx["Open"]).rolling(21).std()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- top panel: volatility estimators ---
spx.loc[:, spx.columns.str.startswith("rv")].plot(ax=ax1)
ax1.set_title("Volatility Estimators")
ax1.set_ylabel("Annualized Volatility")
ax1.grid(alpha=0.3)
ax1.legend()

# --- bottom panel: overnight vs intraday volatility ---
spx[["overnight_volatility", "intraday_volatility"]].plot(ax=ax2)
ax2.set_title("21D Volatility: Overnight vs Intraday")
ax2.set_ylabel("Annualized Volatility")
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# - **Insight:** You’ll see that in stress regimes (e.g., 2008, COVID-19 crash), *overnight moves dominate*. YZ explicitly accounts for this.

# %% [markdown]
# ## 2.3 High-Frequency Estimators: Realized Variance
#
# With **5-minute intraday data**, we can extract much more information about the price path than from daily OHLC bars alone.  
# In this notebook we use **E-mini S&P 500 futures (ES)**, which trade nearly 24 hours a day, so the realized measure naturally captures both **regular-hours and overnight** moves.
#
# A 5-minute grid is widely seen as a good compromise between **information content** and **microstructure noise**: it recovers about **90–95% of daily integrated variance**. At higher frequencies (e.g. 1-minute, tick), microstructure effects (bid–ask bounce, discreteness, etc.) dominate and more sophisticated estimators (e.g. **pre-averaged RV**, **realized kernels**) become necessary.
#
# As our baseline, we use the standard **realized variance (RV)** estimator. Over day $t$, with intraday log-returns $r_{t,i}$,
#
# $$
# RV_t = \sum_{i=1}^{N_t} r_{t,i}^2, 
# \qquad
# r_{t,i} = \ln\!\left(\frac{P_{t,i}}{P_{t,i-1}}\right),
# $$
#
# where $N_t$ is the number of 5-minute intervals in day $t$, and $P_{t,i}$ is the price at intraday time step $i$.

# %%
es_5min = load_intraday_prices(DATA_INTER / "es-5m.csv", start=start, end=end)
es_5min

# %%
# pick 4 random days
all_days = es_5min.index.normalize().unique().date
days = np.random.choice(all_days, size=4, replace=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, day in zip(axes.ravel(), days):
    es_5min.loc[str(day), "close"].plot(ax=ax)
    ax.set_title(day.strftime("%Y-%m-%d"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")

plt.tight_layout()
plt.show()

# %% [markdown]
# There are some interpolated prices, but oeverall teh quality looks good comapred to yfinance for isnatcne where msot of the series would be interpolated.

# %%
daily_rv = rvvol.rv_intraday(es_5min["close"])
es_rv_21 = np.sqrt(daily_rv.rolling(21).mean() * 252)

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

es_rv_21.plot(ax=ax, label="ES 5-min RV", lw=2)
spx.loc[:, spx.columns.str.startswith("rv")].plot(ax=ax, lw=1, alpha=0.8)

ax.set_title("ES vs SPX Realized Volatility")
ax.set_ylabel("Annualized Volatility")
ax.grid(alpha=0.3)
ax.legend()

plt.show()

# %% [markdown]
# The difference between close-to-close and realized variance is evne more pormiennt at daily frequency.

# %% [markdown]
# # **3. Stylized Facts of Daily Volatility**
#
# In this section we investigate several well-established stylized facts of daily volatility that are directly relevant for forecasting. The goal is to highlight key empirical properties of daily variance that can be exploited when building predictive models.
#
# **From a modelling perspective, we work with daily realized variance rather than monthly rolling volatility because:**
# 1. **Variance is additive over time**, while standard deviation is not — making variance a more coherent object for multi-horizon forecasts (e.g., 21-day forward variance).
# 2. The stylized facts documented in the volatility literature (**persistence, clustering, long memory, leverage effects**, etc.) are formulated at the **daily frequency**, not on monthly-smoothed series.
#
# These empirical observations serve as the theoretical motivation for the feature construction and model specifications used throughout the notebook.

# %% [markdown]
# ## 3.1 Volatility clustering
#
# Volatility does not evolve randomly from day to day — instead, it exhibits persistent regimes. Periods of high volatility tend to be followed by high volatility, and periods of low volatility tend to be followed by low volatility.
#
# It implies that recent volatility contains information about future volatility, which justifies the use of lagged realized variance measures (e.g., daily, weekly, monthly RV) as predictors in HAR-type models.

# %%
daily_c2c = spx["returns"].pow(2)

plt.figure(figsize=(12, 6))
daily_rv.loc["2018"].plot(alpha=1, label="Daily Close-To-Close")
daily_c2c.loc["2018"].plot(alpha=0.4, label="Daily Realized Variance")
plt.legend()
plt.show()

# %% [markdown]
# ## 3.2 Long Memory / Slow Decay
#
# Squared returns and realized variance display **very slow autocorrelation decay**, a phenomenon often described as long memory.
# This means today’s volatility contains predictive information not only for the next few days, but even for **several weeks or months ahead.**
#
# This empirical fact motivates the structure of **HAR-RV models**, which include volatility lags at different horizons (daily, weekly, monthly). HAR provides a parsimonious way to approximate long-memory behaviour without requiring a large number of parameters.

# %%
ph.plot_acf_pacf(daily_c2c, lags=40, title="Daily Squarred returns")
ph.plot_acf_pacf(daily_rv, lags=40, title="Daily Realized Variance")

# %% [markdown]
# Here we can see that the ACF of realized varaicn eis much more peristent that the saurred returns and is clearly persistent until the 21-th lags.

# %% [markdown]
# ## 3.3 Mean-reverting behaviour
#
# Volatility is **mean-reverting**: after a shock—whether a volatility spike or a collapse—it tends to slowly drift back toward its long-run unconditional level.
# The reversion is not instantaneous; it occurs gradually over time, reflecting persistent but ultimately temporary deviations from equilibrium.
#
# This property is foundational in volatility modelling and supports the use of **autoregressive structures** (e.g., HAR, GARCH) that capture both persistence and eventual reversion.

# %%
from statsmodels.tsa.ar_model import AutoReg

# Fit AR(1) model
model = AutoReg(daily_rv.to_numpy(), lags=1, old_names=False)
res = model.fit()
print(res.summary())

# %% [markdown]
# The regression coefficient sugguest that the variance has mean-reverting property that can be leveraged in autoregression model like **GARCH**

# %% [markdown]
# ## 3.4 Volatility distributions are often log-normal
#
# Volatility is strictly positive, highly right-skewed, and empirically close to log-normal. In practice, while raw volatility measures (RV, IV) exhibit heavy right tails and large outliers, their logarithm is much closer to a Gaussian distribution.
#
# **Why log-volatility is preferred**
# - **Positivity:** Modeling log-volatility ensures that forecasts, once exponentiated, are always ≥ 0.
# - **Statistical convenience:** Log-volatility has a distribution much closer to Normal → residuals from regressions are more symmetric and homoscedastic.
# - **Better predictive behaviour:** Many linear and autoregressive models (HAR, ARMA, regressions) fit significantly better on log-transformed volatility.
#
# These properties make the log-volatility transformation a standard step in volatility forecasting models. 

# %%
ph.plot_hist_transform(daily_rv, use_log=True, figsize=(10,4))

# %% [markdown]
# ## 3.5 Leverage Effect / Asymmetric Return–Volatility Relationship
#
# Another key stylized fact is the asymmetric relationship between returns and volatility:
# - **Negative returns** (price drops) tend to increase future volatility more than positive returns of the same magnitude.
# - This phenomenon is known as the **leverage effect**: when the equity price falls, financial leverage rises, making the firm riskier and increasing volatility.
# - Empirically, this produces an asymmetric correlation:
#   
# $$\operatorname{Corr}(r_t,\ \sigma^2_{t+h}) < 0.$$
#
# **Modeling implications**
# - **Symmetric** volatility models (e.g., standard GARCH) cannot capture this effect.
# - **Asymmetric / leverage-sensitive** models—EGARCH, GJR-GARCH, Heston with leverage, etc.—are designed to incorporate it.
# - For **RV forecasting**, include **downside-biased return measures** (e.g., negative returns, downside semivariance) as predictive features.

# %%
fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# SPX closing price
ax1 = fig.add_subplot(gs[0])
ax1.plot(spx.index, spx["Close"], color="steelblue")
ax1.set_title("SPX Closing Price")
ax1.set_ylabel("Price")

# realized volatility
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(daily_rv.index, daily_rv, color="darkorange", linestyle="--")
ax2.set_title("Daily Realized Variance")
ax2.set_ylabel("Variance")
ax2.set_xlabel("Date")

plt.tight_layout()
plt.show()

# %% [markdown]
# Here we can se ethat big drops occuring arround 2012, 2016 and 2020 covid crash are associated with higher vraaince

# %% [markdown]
#
# ---
#
# # **4. Problem Formulation**
#
# We frame the task as a supervised learning problem: forecasting the 21-day forward realized volatility of the S&P 500.
#
# Formally, the objective is to predict forward realized volatility using only information available at time $t$.  
#
# $$
# y_t = f(X_t; \beta) + \varepsilon_t, \quad \text{with } \mathbb{E}[y_t | X_t] = f(X_t; \beta)
# $$
#
# where:
# - $y_t$ = target variable (forward 21-day realized volatility),
# - $X_t$ = vector of predictors at time $t$.
# - $f(.;\beta)$ = parametric or non-parametric form of the regressors
#

# %% [markdown]
# ## 4.1 Target Variable: 21-Day Realized Variance
#
# For forecasting purposes, it is convenient to define the target in terms of **daily realized volatility** averaged over the forward horizon. Since realized variance and volatility are highly skewed and approximately log-normal, we work in the **logarithmic scale**:
#
# $$
# y_t = \log\!\big(RV_{t+1:t+21}\big)
# $$
#
# where the forward 21-day realized variance aligned at time $t$ is defined as:
#
# $$
# RV_{t+1:t+21} 
# = \frac{RV_{t+1} + RV_{t+2} + \cdots + RV_{t+21}}{21}
# $$
#
# with $RV_{t+i}$ denoting the one-day realized volatility computed from intraday (5-min) returns.
#
# - The **log transform** reduces skewness, stabilizes variance, and makes the distribution closer to Gaussian.  
# - It also guarantees positivity when transformed back:
#
# $$
# \hat{RV}_{t+1:t+21} = \exp(\hat{y}_t).
# $$

# %%
H = 21
y = rvfeat.create_forward_target(daily_rv, horizon=H)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

np.exp(y).hist(bins=30, ax=axes[0], label="RV", color="steelblue", alpha=0.7)
axes[0].set_title("Distribution of 21-D Realized Variance")
axes[0].legend()

y.hist(bins=30, ax=axes[1], label="Log(RV)", color="darkorange", alpha=0.7)
axes[1].set_title("Distribution of Logarithm of 21-D Realized Variance")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.2 Predictor Families (Information Available at Time $t$)
#
# We use only information observable at time $t$, grouped into predictor families, where each family captures a different aspect of future (forward) RV.
#
# We organise the features into the following blocks:
# - **Lagged volatility measures** (daily/weekly/monthly RV)
# - **IV surface predictors** (level, slope, skew)
# - **Returns-based predictors** (realised returns, downside vs upside moves, overnight returns)
# - **Macro & Market Preidctors** (rates, term spread, credit spreads, VIX, VVIX)
#
# For each predictor family we:
# - plot the **time series** to see regimes and structural breaks,
# - look at the **distribution** to decide on transformations (log, sqrt, winsorisation),
# - and inspect **relationship with the target** via scatter / hexbin plots.

# %% [markdown]
# ### 4.2.1 Lagged Volatility Measures: HAR-RV Lags (1D, 5D, 22D)
#
# As seen in the ACF plots, realized variance displays strong persistence, remaining significant over many daily lags. This motivates using **lagged realized variance** as predictors.
#
# We adopt the standard **HAR-RV (Heterogeneous Autoregressive)** structure, which captures short-, medium- and long-term memory via daily, weekly and monthly averages:
#
# $$
# X_{t}^{HAR} = RV_{D,t} + RV_{W,t} + RV_{M,t}
# $$
#
# where:
# - $RV_{D,t} = RV_{t}$ (daily lag, short-term persistence)  
# - $RV_{W,t} = \tfrac{1}{5}\sum_{i=0}^{4} RV_{t-i}$ (weekly average, medium-term)  
# - $RV_{M,t} = \tfrac{1}{21}\sum_{i=0}^{20} RV_{t-i}$ (monthly average, long-term)  
#
# This specification smooths noisy daily lags and reflects the **heterogeneous horizons** of market participants (daily traders, weekly rebalancers, monthly institutions).

# %%
summary_stats = ['count','mean','std','min','max','skew','kurtosis']

X_har = rvfeat.create_har_lags(daily_rv)
X_har.agg(summary_stats)

# %%
fig, ax = plt.subplots(1,1, figsize=(12, 5))
ax.set_yscale("log")
X_har["RV_D"].plot(ax=ax, alpha=0.4, label="Log (RV_D)")
X_har["RV_W"].plot(ax=ax, alpha=1, label="Log (RV_W)")
X_har["RV_M"].plot(ax=ax, alpha=1, label="Log (RV_M)")
plt.legend();
plt.title("HAR-RV Log components")
plt.show()

# %% [markdown]
# The HAR-RV lags evolve closely together over time, but each one captures a different volatility regime:
# short-term (1-day), medium-term (5-day), and long-term (21-day/monthly).

# %%
ph.plot_feature_histograms(X_har, figsize=(10, 3))

# %% [markdown]
# #### **Transformation Decisions**
# - Apply Log transform to: `RV_D`, `RV_W`, `RV_M`.

# %%
ph.plot_features_vs_target(X_har, y, log_features=X_har.columns.tolist(), figsize=(10, 3), nrows=1, ncols=3)

# %% [markdown]
# Each predictor shows a **clear linear relationship** with the target in the log space, which helps explain why the **HAR-RV model is so popular** and widely used as a benchmark.

# %% [markdown]
# ### 4.2.2 Implied Volatility Surface Predictors 
#
# Realized volatility lags capture **persistence**, while implied volatility (IV) carries the market’s **forward-looking expectations** and risk premia. We use:
#
# - **ATM IV (1M)**  
#   Near-the-money 1-month IV, tightly linked to option premia and cleaner than broad indices like VIX.
#
# - **25Δ IV Skew (Risk Reversal)**  
#   Captures the **downside risk premium** and crash-protection demand: steeper (more negative) skew ↔ stronger tail-risk pricing.
#
# - **IV Term Structure**  
#   Front-loaded (short > long) IV usually signals near-term stress; an upward slope is consistent with calmer, mean-reverting regimes.
#
# These IV predictors complement HAR-RV lags by adding a **market-implied view of future volatility**.

# %% [markdown]
# ### SPX option chain
#
# We first load the SPX option chain and add basic surface coordinates: time-to-maturity $T$ and log-moneyness $k = \log(K/S)$.

# %%
spx_options = pd.read_parquet(DATA_INTER / "full_spx_options_2010_2020.parquet")
spx_options["T"] = spx_options["dte"] / 252 
spx_options["k"] = np.log(
    spx_options["strike"] / spx_options["underlying_last"]
 )
spx_options.head()

# %% [markdown]
# ### SSVI-based IV surface predictors
#
# We then calibrate an SSVI surface slice-by-slice and extract a small set of IV-based predictors (ATM level, skew, term structure, etc.) aligned to the RV target dates.

# %%
ssvi = SSVI()
df_g = pd.read_parquet(DATA_PROC / "ssvi_globals_2010_2020.parquet")
df_k = pd.read_parquet(DATA_PROC / "ssvi_knots_2010_2020.parquet")

params_ssvi = ssvi.build_params_dict(df_g, df_k)
X_iv_surface = rvfeat.create_iv_surface_predictors(spx_options, ssvi, params=params_ssvi)
X_iv_surface = X_iv_surface.reindex(y.index).ffill()

X_iv_surface.agg(summary_stats)

# %% [markdown]
# ### Correct data error from iv puts
#
# A few IV-surface days are clearly erroneous / inconsistent (incorrect data error from IV put). We set those dates to NaN and linearly interpolate to avoid contaminating the predictors.

# %%
X_iv_surface.loc["2019-05-02":"2019-05-08", :] = np.nan
X_iv_surface.loc["2019-05-17", :] = np.nan
X_iv_surface = X_iv_surface.interpolate(method="linear")

X_iv_surface.agg(summary_stats)

# %%
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green, red
fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

for ax, col, c in zip(axes.flat, X_iv_surface.columns, colors):
    X_iv_surface[col].plot(ax=ax, lw=1.5, color=c)
    ax.set_title(col, fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("IV")
    
plt.tight_layout()
plt.show()

# %% [markdown]
# We see that during market stress, **ATM IV spikes**, the **skew steepens**, and the **IV term structure inverts** (front-month IV ≫ back-month IV).

# %%
ph.plot_feature_histograms(X_iv_surface, figsize=(10, 3))

# %% [markdown]
# #### **Transformation Decisions**
# - Apply Log transform to: `iv_atm_60`, `iv_skew`.
# - Apply Winsorization to: `iv_ts`

# %% [markdown]
# During market crashes the IV spike, skew steepens, the Iv-term strcutrue gets inverted.

# %%
ph.plot_features_vs_target(X_iv_surface, y, log_features=["atm_iv_30d", "iv_skew"], figsize=(10, 3), nrows=1, ncols=3)

# %% [markdown]
# Here, a **higher ATM IV level** and a **steeper skew** are associated with **higher forward 21-day realized variance**, whereas the relationship is **negative for the IV term structure** (more inverted term structure → higher future RV).

# %% [markdown]
# ### 4.2.3 Return-Based Predictors  
#
# Realised volatility is tightly linked to the behaviour of **recent returns**, but raw daily returns are very noisy.  
# Since our target is **21-day RV**, we summarise returns over **short rolling windows** (e.g. 5 trading days) to extract more stable signals:
#
# - **Volatility clustering / persistence**  
#   Rolling averages of **absolute** and **squared** returns capture the current volatility regime (calm vs stressed).
#
# - **Asymmetry / leverage effect**  
#   Large **negative** returns tend to increase future volatility more than positive ones of the same magnitude. We proxy this via downside-focused measures (e.g. rolling sums of negative returns).
#
# - **Jumps / overnight moves**  
#   Large overnight gaps ($Open_t$ - $Close_{t−1}$) often reflect news shocks and regime shifts, and are typically followed by higher RV.
#
# - **Shape of the recent return distribution**  
#   Rolling **realised skewness** and **kurtosis** capture asymmetry and tail thickness of recent returns, both of which are associated with stressed markets.
#
# These return-based predictors complement HAR-RV lags and IV-based features by encoding **recent realised price dynamics**.

# %%
X_returns = rvfeat.create_return_predictors(spx["returns"], es_5min, h=H)
X_returns = X_returns.reindex(y.index).ffill()
X_returns.agg(summary_stats)

# %%
ph.plot_feature_histograms(X_returns, figsize=(10, 4))

# %% [markdown]
# #### **Transformation Decisions**
# - Apply Log transform to: `abs_r`, `r2`, `down_var`, `up_var`
# - Apply Winsorization transform to: `overnight_ret`
# - Apply Sqrt + Winsorization transform to: `neg_r2`

# %%
log_features_ret = ["abs_r", "r2", "down_var", "up_var"]
sqrt_features_ret = ["neg_r2"]
ph.plot_features_vs_target(X_returns, y, log_features=log_features_ret, sqrt_features=sqrt_features_ret, figsize=(12, 5), nrows=2, ncols=4)

# %% [markdown]
# ### 4.2.4 Macro & Market Predictors  
#
# This block mixes **macro fundamentals** (rates, credit spreads) with **market-based indicators** (VIX and VVIX).  
# On their own they are usually **weaker forecasters** than RV lags or IV signals, but they help capture **regime shifts** and can be exploited by more flexible models (e.g. RF).

# %% [markdown]
# #### Macro Fundamentals  
#
# We include **slow-moving economic and financial variables** such as:
#
# - **Interest rate levels & term spreads**: 3M / 2Y / 10Y Treasury yields, yield-curve slope.  
# - **Credit spreads**: IG and HY option-adjusted spreads as proxies for systemic stress.
#
# These variables do not explain day-to-day volatility, but they are useful to identify **macro regimes**  
# (e.g. tight monetary policy, widening credit spreads) that can change the level and persistence of forward RV.

# %%
X_macro = rvfeat.create_macro_features(start=start, end=end)
X_macro = X_macro.reindex(y.index).ffill()
X_macro.agg(summary_stats)

# %%
ph.plot_macro_block(X_macro)

# %% [markdown]
# The widening of the credit spreads line up with stress episodes (Euro crisis, 2015–16, Covid), which are also periods of elevated forward RV.

# %%
ph.plot_feature_histograms(X_macro, figsize=(8, 4), nrows=2, ncols=3)

# %% [markdown]
# #### **Transformation Decisions**
# - No data transformation is needed (of course scaling for linear models)

# %%
ph.plot_features_vs_target(X_macro, y, log_features=["HY_OAS", "IG_OAS"], figsize=(10, 5), nrows=2, ncols=3)

# %% [markdown]
# In the post-GFC low-rate regime, the 3-month Treasury yield (DGS3MO) is **essentially stuck near zero** up to about 2017, so it carries very little **independent variation**. Since its information is already **embedded in the term-spread variable** term_spread_10y_3m, we drop DGS3MO and keep the spread as the macro predictor.

# %%
X_macro = X_macro.drop("DGS3MO", axis=1)

# %% [markdown]
# #### Market / Sentiment  
#
# We also include **market-implied risk indicators** that move faster than macro variables and help identify **risk-on / risk-off regimes**:
#
# - **VIX** – 30-day S&P 500 implied volatility; a fast barometer of market fear and near-term uncertainty.  
# - **VVIX** – “vol of vol”; measures uncertainty about future changes in VIX and often spikes in unstable volatility regimes.  
# - **VIX term structure** – the slope between VIX and VIX3M (or front vs back VIX futures), which distinguishes **short-term stress** (inverted term structure) from calmer, upward-sloping regimes.  
#
# These do not describe the asset’s volatility directly, but provide **high-frequency information on risk appetite and regime**, useful as **conditioning variables** for IV–RV forecasts.

# %%
X_market = rvfeat.create_market_features(start=start, end=end)
X_market = X_market.reindex(y.index).ffill()

# %%
ph.plot_feature_histograms(X_market, figsize=(8, 3))

# %% [markdown]
# #### **Transformation Decisions**
# - Apply `log` transform to: `VIX`, `VVIX`
# - Apply Winsorization transform to: `vix_ts`

# %%
ph.plot_features_vs_target(X_market, y, log_features=["VIX", "VVIX"], figsize=(10, 3), nrows=1, ncols=3)

# %% [markdown]
# # **5. Feature Engineering**
#
# We go beyond raw predictors (HAR lags, VIX, IV surface, credit spreads, etc.) by creating features that are more aligned with the forecasting task. The goal is to reduce noise, capture regime levels, and detect stress transitions that precede volatility spikes.  
#
# - **Horizon-aligned smoothing:** The target is 21-day forward realized variance, so daily predictors are too noisy. We create regime-level signals that capture persistent volatility regimes instead of one-day noise.  
#
# - **Regime shift / stress dynamics:** Volatility surges usually follow abrupt risk repricing. To capture this we add regime change features like momentum in VIX and SKEW as well as intercation effects like VVIX and VIX.

# %%
X_core = pd.concat([X_har, X_iv_surface, X_macro, X_returns, X_market], axis=1)
X_eng = rvfeat.feature_engineering(X_core)
X = pd.concat([X_core, X_eng], axis=1)
X = X.dropna(axis=0)

core_features = X_core.columns.tolist()
eng_features = X_eng.columns.tolist()

X_eng.agg(summary_stats)

# %%
ph.plot_feature_histograms(X_eng, figsize=(10, 5))

# %%
log_features_eng = list(set(eng_features) - set(["dVIX_5d", "dSkew_5d", "vvix_over_vix"]))
ph.plot_features_vs_target(X_eng, y, log_features=log_features_eng, figsize=(10, 6), nrows=3, ncols=4)

# %%
del X_har, X_iv_surface, X_macro, X_returns, X_market, X_core, X_eng
gc.collect()

# %% [markdown]
# # **6. Data Preprocessing**
#
# We inspect each predictor’s distribution (shape, skewness, tail heaviness) and only transform **when necessary**.  
# The goal is to make features **more symmetric** and **well-behaved** for regression and machine learning.
#
# We rely on simple, interpretable transforms:
#
# - **log** for strictly positive, heavy-right-tailed variables (e.g. VIX, kurtosis proxies),
# - **sqrt** for variance-like or squared quantities (e.g. squared returns),
# - **winsorization** (clipping extreme quantiles) to tame rare but extreme outliers.
# - **scaling** apply standard scaling to put all the features on the same scale (essential for linear models)
#

# %%
ph.plot_hist_transform(X["iv_skew"], use_log=True)
ph.plot_hist_transform(X["iv_ts"], winsorize=(0.005, 0.995))
ph.plot_hist_transform(X["neg_r2"], use_sqrt=True)

# %% [markdown]
# ## Apply the appropiate transformation
#
# We apply all transformations using the `DataProcessor` class, which follows the scikit-learn **Transformer / Pipeline** API:
# we **fit** the transformations only on the training set, then **apply** them to both the training and test data.

# %%
log_features = [
    "RV_D", "RV_W", "RV_M", "RV_D_ewma",
    "abs_r", "r2", "down_var", "up_var",
    "VIX", "VVIX", "VIX_rm5", "VIX_rm21", "VIX_ewma",
    "atm_iv_30d", "iv_skew", "iv_minus_realized",
    "VIX_time_HY_OAS", "RV_D_rollvol5", "RV_D_rollvol21"
]

winsor_sqrt_features = [
    ([0.0, 0.995], ["neg_r2"])
]

winsor_features = [
    ([0.01, 1],      ["vix_ts"]),
    ([0.005, 0.995], ["iv_ts", "overnight_ret"]),
    ([0.001, 0.995], ["dVIX_5d", "dSkew_5d"])
]

dp = DataProcessor(
    log_features=log_features,
    winsor_sqrt_features=winsor_sqrt_features,
    winsor_features=winsor_features,
    scale=True
)

X_proc = dp.fit_transform(X)
X_proc = pd.DataFrame(X_proc, index=X.index, columns=dp.get_feature_names_out())

rng = np.random.default_rng(42)
cols_sample = rng.choice(X_proc.columns, size=8, replace=False)
X_proc[cols_sample[:4]].agg(summary_stats)

# %%
ph.plot_feature_histograms(X_proc[cols_sample], figsize=(10, 4))

# %% [markdown]
# ### Data preprocessing pipeline
#
# The figure below shows the full `DataProcessor` pipeline: log transforms, winsorisation, optional sqrt transform, and final scaling.  
# This makes the feature engineering and preprocessing **explicit and reproducible**.
#
# ![data_processor.png](attachment:data_processor.png)

# %%
y = rvfeat.create_forward_target(daily_rv, horizon=21)

data = pd.concat([X_proc, y], axis=1)
data = data.dropna()

X_clean = data[X_proc.columns]
y_clean = data[y.name]

# %% [markdown]
# ## Correlation / redundancy check
#
# Here we remove **highly correlated features** so that feature-importance scores in the next section are **more stable and easier to interpret**.
# We use a **conservative threshold** of $\lvert \rho \rvert = 0.95$ to avoid over-pruning the feature set, since both Lasso and Random Forest are relatively robust to moderate multicollinearity.

# %%
corr = X_clean.corr()

plt.figure(figsize=(8, 6)) 
sns.heatmap(
    corr,
    cmap="coolwarm",        
    center=0,                   
    cbar_kws={"shrink": .8}
)
plt.title("Feature Correlation Heatmap", fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# %%
corr_threshold = 0.95
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_pairs = (
    upper.stack()
         .loc[lambda s: s > corr_threshold] 
         .sort_values(ascending=False)
)
print(high_pairs)

# %% [markdown]
# We decide which columns to drop by keeping the **original, best-documented features**. For example, we keep `VIX` and drop `atm_iv_30d`: they carry very similar information, but **VIX is more widely documented** in the literature as a meaningful predictor.

# %%
corr_features_to_drop = [
    "atm_iv_30d", "iv_minus_realized", "r2",
    "VIX_rm5", "RV_D_ewma", "HY_OAS", "VIX_ewma", "RV_D_rollvol21"
]
X_clean = X_clean.drop(corr_features_to_drop, axis=1)

# %% [markdown]
#
# ---
#
# # 7. **Modelling Framework**

# %% [markdown]
# ## 7.1 Cross-Validation Scheme (Purged K-Fold)
#
# To evaluate feature importance **across time**, we need a CV scheme that respects the time-series structure and avoids look-ahead.
#
# We use a **Purged K-Fold** cross-validation:
#
# - Folds are **contiguous in time** (no shuffling).
# - Around each test fold, we **purge** neighbouring observations to remove overlap from:
#   - the 21-day RV target horizon, and
#   - rolling windows used in the predictors.
# - We add a small **embargo** after the test fold to avoid subtle leakage from nearby points.
#
# This produces a sequence of non-overlapping validation blocks over 2010–2020, allowing us to study **feature importance and stability across regimes** while controlling for target overlap.

# %%
purged_cv = PurgedKFold(
    n_splits=5,      # ≈ 2-year validation blocks
    purge_gap=21,   # 21-day RV horizon
    embargo=0.01,  # 1% embargo after validation fold
)

ph.plot_purged_kfold_splits(purged_cv, X_clean, y_clean)

# %% [markdown]
# ## 7.2 Model Specifications  
#
# Building on the volatility-forecasting literature and our exploratory analysis, we use:
#
# - **Linear regression / HAR-type models** as the main econometric benchmark,  
# - **Random Forests** as a flexible non-linear alternative.
#
# The modelling path is:
#
# - start from the canonical **HAR-RV** specification,  
# - extend to **HAR-RV-X** by adding a small set of predictors with robust predictive power and clear economic meaning  
#   (IV level, term structure, VRP, overnight moves, macro spreads, etc.),  
# - fit a **Random Forest** on the same feature set and compare its performance and behaviour to the linear models.

# %% [markdown]
# ### 7.2.1 Linear Models (HAR-RV and HAR-RV-X)  
#
# We consider a baseline **HAR-RV** specification of the form:
#
# $$
# \text{RV}^{(21)}_{t+21}
# = \beta_0 
# + \beta_D \,\text{RV}^{(1)}_t
# + \beta_W \,\text{RV}^{(5)}_t
# + \beta_M \,\text{RV}^{(21)}_t
# + \varepsilon_{t+21},
# $$
#
# where $\text{RV}^{(k)}_t$ denotes a $k$-day average of past daily realized variance (HAR daily/weekly/monthly structure).
#
# We then extend this to a more general **HAR-RV-X** model:
#
# $$
# \text{RV}^{(21)}_{t+21}
# = \beta_0 
# + \beta_D \,\text{RV}^{(1)}_t
# + \beta_W \,\text{RV}^{(5)}_t
# + \beta_M \,\text{RV}^{(21)}_t
# + \gamma^\top X_t
# + \varepsilon_{t+21},
# $$
#
# where $X_t$ collects additional predictors (IV level, IV–RV spread, term structure, skew, overnight returns, macro/sentiment variables, etc.) that are selected based on feature-importance and stability analysis in Section 8.

# %%
lin_model = LinearRegression()

# %% [markdown]
# ### 7.2.2 Non-Linear Model (Random Forest)  
#
# As a non-linear benchmark we use a **Random Forest regressor**:
#
# $$
# \widehat{\text{RV}}^{(21)}_{t+21}
# = f_{\text{RF}}(X_t),
# $$
#
# where $f_{\text{RF}}$ is an ensemble of decision trees fitted on the same feature vector $X_t$.  
#
# Random Forests can capture **non-linearities** and **interaction effects** (e.g. regime-dependent impact of VIX, IV term structure, or macro variables) that are not explicitly modelled in the linear HAR-RV-X specification. 

# %%
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

# %% [markdown]
# ## 7.3 Evaluation Metrics
#
# We evaluate forecasts on the log-RV target using:
#
# - **R² (sklearn):**  
#   Standard out-of-sample coefficient of determination:
#   $$
#   R^2 = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \bar y)^2}.
#   $$
#
# - **MSE** and **QLIKE:**  
#   Mean squared error on log-RV, and a log-loss defined on the variance scale:
#   $$
#   QLIKE = \mathbb{E}\Big[\frac{\hat\sigma_t^2}{\sigma_t^2}
#                  - \log\frac{\hat\sigma_t^2}{\sigma_t^2} - 1\Big],
#   $$
#   where $\sigma_t^2 = e^{y_t}$, $\hat\sigma_t^2 = e^{\hat y_t}$.
#
# - **Out-of-sample $R^2$ relative to a benchmark:**  
#   Following the forecasting literature, we also measure the gain vs a
#   benchmark forecast $\hat y^{bench}_t$ (e.g. naive RV or HAR-RV):
#   $$
#   R^2_{OOS}
#   = 1 - \frac{\sum_t (y_t - \hat y^{model}_t)^2}
#                {\sum_t (y_t - \hat y^{bench}_t)^2}.
#   $$
#   By construction, the benchmark itself has $R^2_{OOS} = 0$, and
#   positive values indicate a reduction in MSE vs the benchmark.

# %% [markdown]
# # **8. Feature Importance and Stability Analysis**
#
# The goal of this section is to identify **which features matter** for 21-day RV and **how stable** their contribution is across time.
#
# We combine several importance measures, both **in-sample** and **out-of-sample**, for **linear** and **non-linear** models to:
#
# - confirm core predictors (RV lags, VIX / IV),
# - gauge the usefulness of weaker signals (macro, sentiment, flow),
# - remove clearly irrelevant or redundant variables.
#
# For the **linear HAR-RV-X specifications** we are deliberately strict: the aim is a small, interpretable model.  
# A feature is a candidate for inclusion only if it satisfies all of:
#
# 1. **Economic sense** — clear story and expected sign.  
# 2. **Coefficient stability** across folds — non-zero and sign-stable.  
# 3. **Out-of-sample predictive power** — permutation importance clearly above noise.  
# 4. **Non-redundancy** — not just a near-duplicate of an existing core feature (e.g. another clone of `RV_M` or `VIX`).

# %% [markdown]
# ## 8.1 Single-Feature Importance (SFI)
#
# SFI measures **explanatory power one variable at a time**: for each feature, we fit a simple model $y \sim x_j$ and evaluate performance across validation folds. We use it as a **quick diagnostic for linear models** to see which predictors have **stand-alone signal** for 21-day RV.
#
# It is naturally immune to **substitution effects** (multicollinearity), since no competing features are present, but it **ignores interactions and joint effects** — a feature that is weak alone may still be useful in combination — so we use SFI only as a **complementary tool**, not as the sole basis for feature selection.

# %%
sfi_df, scores = single_feature_importance(X_clean, y_clean, lin_model, purged_cv)
sfi_df.head(10)

# %% [markdown]
# ## 8.2 In-model stability analysis (with substitution effects)
#
# Here we look at **in-sample explanatory power** in a multivariate setting, where substitution effects (multicollinearity) can appear.
#
# Features that are consistently shrunk to (or near) zero in-sample are treated as having **no meaningful explanatory power** and are candidates for removal. If a variable cannot explain the target even in-sample, any apparent predictive power is likely spurious.
#
# ### Lasso
#
# For Lasso we monitor:
# - the **average coefficient** and its **standard deviation** across CV folds,
# - whether the **sign of each coefficient is stable** (no random sign flips).
#
# Because Lasso depends on the regularisation parameter $\alpha$, we use **LassoCV** to select the $\alpha$ that maximises validation performance, and then interpret the coefficients at that chosen level of sparsity.

# %%
lasso_cv = LassoCV(
        alphas=np.logspace(-4, 0, 30),
        cv=purged_cv,
        max_iter=100000,
        n_jobs=-1
)
lasso_cv.fit(X_clean, y_clean)
lasso = Lasso(alpha=lasso_cv.alpha_)

# %%
lasso_coefs, lasso_summary = in_sample_stability(X_clean, y_clean, lasso, purged_cv)

# %%
ph.plot_mean_std_importance(
    lasso_summary,
    title="Lasso coefficients (mean ± std)",
    top_n=20,
    abs_values=True,
)

# %%
ph.plot_lasso_coef_paths(lasso_coefs, lasso_summary["feature"])

# %% [markdown]
# - `VIX_time_HY_OAS` is a **regime-specific** candidate: it only becomes meaningful in the Covid / high-rate period and is strongly correlated with VIX (ρ ≈ 0.94). Thus, for the linear model we decide to drop it, as it introduces substitution effects.
#
# - We keep the other features for now and assess their **out-of-sample** importance across folds.

# %%
lasso_features = lasso_summary.loc[lasso_summary["mean"].abs() > 0.0, "feature"]
X_clean_lin = X_clean[lasso_features]
X_clean_lin = X_clean_lin.drop(["VIX_time_HY_OAS"], axis=1)

# %% [markdown]
# ### Random Forest
#
# We use the standard **Mean Decrease in Impurity (MDI)** importance computed per fold. Features with consistently negligible MDI are treated as **completely useless** and can be dropped from the RF feature set.

# %%
rf_fi, rf_summary = in_sample_stability(X_clean, y_clean, rf, purged_cv)

# %%
ph.plot_mean_std_importance(
    rf_summary,
    title="RF Mean Decrease Impurity (mean ± std)",
    top_n=len(rf_summary),
    abs_values=True,
)

# %% [markdown]
# All features are selected in-sample because they reduce impurity, so they are **all explanatorily relevant**. We therefore keep the full set, including lower-importance variables like `dVIX` and `dSkew`, which may still matter in **stress regimes** when combined with other predictors.

# %% [markdown]
# ## 8.3 Out-of-Sample Permutation Importance (Mean Decrease Accuracy)
#
# We measure each feature’s **marginal OOS contribution**. For every CV split:
#
# - 1. **Fit** the model on the training block,  
# - 2. **Permute** one feature in the **validation** block,  
# - 3. Record the **performance drop** (Δloss = permuted loss − baseline loss).
#
# With correlated predictors, PI can **understate** a feature’s value (another correlated feature “covers” for it) and may discard variables that look useful in-sample. Therefore, interpret PI together with **In-model stability** and **Economic priors**.

# %% [markdown]
# ### Linear Regression

# %%
lasso_pi, lasso_pi_summary = oos_perm_importance(
    X_clean_lin, y_clean, lin_model, purged_cv, random_state=42
)

# %%
ph.plot_mean_std_importance(
    lasso_pi_summary,
    title="Linear OOS Permutation Importance (mean ± std)",
    top_n=20,
    abs_values=False,
)

# %% [markdown]
# - `VVIX` shows **good OOS importance** and clear **in-sample relevance**. Even though its coefficient is materially non-zero in only one fold and essentially zero in the others, we still keep it. It is **strongly justified economically**, and we prefer it over `vvix_over_vix`, since both carry similar information but `vvix_over_vix` has much more unstable OOS performance (larger cross-fold std).
#
# - `iv_ts` has **stable coefficients**, a **consistent negative sign** (economically reasonable for an inverted IV term structure), **moderate OOS importance**, and low correlation with other predictors, so we keep it as a candidate for HAR-RV-X.
#
# - `neg_r2` is also stable across folds, shows **clear predictive power**, and has a **strong economic rationale** (captures the **leverage effect**), so we include it in HAR-RV-X.
#
# - Other features fail at least one of our criteria (unstable sign or near-zero coefficients, no clear OOS importance, or weak economic story) and are not retained in the parsimonious linear specification.

# %% [markdown]
# ### Random Forest

# %%
rf_pi, rf_pi_summary = oos_perm_importance(
    X_clean, y_clean, rf, purged_cv, random_state=42
)

# %%
ph.plot_mean_std_importance(
    rf_pi_summary,
    title="RF OOS Permutation Importance (mean ± std)",
    top_n=len(rf_pi_summary),
    abs_values=False
)

# %% [markdown]
# We prune a small set of clearly redundant / weak RF features that show no meaningful importance in or out-of-sample:
#
# - `rku`: slight in-sample importance, very neagtive OOS signal
# - `term_spread_10y_3m`: slight in-sample importance, no OOS signal, weak economic story  
# - `VIX_rm21`: highly correlated with `RV_M` (ρ ≈ 0.92), no extra OOS signal  
# - `DGS2`: unstable and insignificant OOS importance, no strong econ link to 21D RV  
# - `RV_D_rollvol5`: no clear IS/OOS signal, strongly related to `RV_W`  
# - `dSkew_5d`: unused IS/OOS (junk feature)  
# - `overnight_ret`: no IS importance, small negative OOS contribution  
# - `up_var`: dominated by `down_var` for leverage / downside effects

# %%
features_to_drop = [
    "rku",
    "rsk",
    "term_spread_10y_3m",
    "VIX_rm21",
    "DGS2",
    "up_var",
    "RV_D_rollvol5",
    "dSkew_5d",
    "overnight_ret"
]

X_clean_rf = X_clean.drop(columns=features_to_drop)
features_rf = X_clean_rf.columns
print(features_rf)

# %% [markdown]
# # **9. Model Selection**
#
# We now choose the **final fixed models** to carry into the truly out-of-sample walk-forward backtest. All comparisons use the same **Purged K-Fold** scheme and identical preprocessing for a fair, like-for-like evaluation.
#
# We compare three layers:
# - two simple **benchmarks** (Naive RV, Naive IV),
# - a family of **HAR-RV-X linear models**,
# - a tuned **Random Forest** as a non-linear benchmark.
#
# The main selection criterion is OOS CV performance (R² / MSE / QLIKE), with a **parsimony bias**: when models are statistically close, we prefer the **simpler** one (e.g. HAR-type over RF).
#
# Finally, we compare the best linear and RF models and test simple linear ensembles to see whether combining them **improves accuracy and/or reduces error variance**. If ensembles do not add value, we retain the single best parsimonious model.

# %%
data = pd.concat([X, y], axis=1)
data = data.dropna()

X_train = data[X.columns]
y_train = data[y.name]

purged_cv = PurgedKFold(
    n_splits=10,      # ≈ 1-year validation blocks
    purge_gap=21,   # 21-day RV horizon
    embargo=0.01,  # 1% embargo after validation fold
)

# Common data preprocessing config
dp_kwargs = dict(
    log_features=log_features,
    winsor_sqrt_features=winsor_sqrt_features,
    winsor_features=winsor_features
)

# %% [markdown]
# ## 9.1 Benchmarks (Naive RV and Naive IV)
#
# We use **two complementary benchmarks**, each playing a different role:
#
# - a **Naive RV** benchmark → *statistical floor* for any forecasting model,  
# - a **Naive IV** benchmark → *economic / trading benchmark* for IV–RV strategies.
#
# **Naive RV (persistence benchmark)**  
# This uses the last observed 21-day realized variance as the forecast of the next 21-day variance:
# $$
# \hat y^{\text{NaiveRV}}_t = \log\big(\text{RV}^{(21)}_{t}\big).
# $$
# Any model that cannot beat this simple persistence rule has **no real forecasting value**.
#
# **Naive IV (ATM 30D benchmark)**  
# Here we use the 30-day ATM implied volatility as a direct forecast of future daily variance:
# $$
# \hat y^{\text{NaiveIV}}_t
# = \log\Big(\frac{\text{IV}_{30\text{D},t}^2}{252}\Big),
# $$
#
# This benchmark represents the **market’s own forecast** of future volatility.
# Beating Naive IV implies a **forecasting edge** (our model is closer to realized RV), but only a **potential trading edge** — profitability still depends on execution, costs, and risk management in the IV–RV strategy

# %%
y_naive_rv, y_naive_iv = rvfeat.build_naive_targets(
    X_train["RV_M"], X_train["atm_iv_30d"]
)

metrics_rv = compute_metrics(y_train, y_naive_rv)
metrics_iv = compute_metrics(y_train, y_naive_iv, y_naive_rv)

row_rv = {"model": "Naive-RV", **metrics_rv}
row_iv = {"model": "Naive-IV", **metrics_iv}

metrics_bench = pd.DataFrame([row_rv, row_iv]).set_index("model")
display(metrics_bench)

# %% [markdown]
# ## 9.1 Linear Models
#
# We compare three nested HAR-type specifications:
#
# 1. **HAR-RV**  
#    The classical benchmark that uses only realized volatility lags to forecast future RV.  
#    This specification is well established in the volatility-forecasting literature.
#
# 2. **HAR-RV-VIX**  
#    A natural extension that augments HAR-RV with a forward-looking volatility measure, the VIX.  
#    Several studies document VIX as a strong predictor of 1-month realized volatility.
#
# 3. **HAR-RV-VIX-X**  
#    A further extension where we add a small set of economically motivated predictors
#    $X_t$ that show robust predictive power for forward RV.  
#    In our case we consider:
#    - **HAR-RV-VIX-VVIX**: adding the VVIX as a forward looking vol-vol predictor,
#    - **HAR-RV-VIX-L**: adding a leverage-effect proxy (downside returns),
#    - **HAR-RV-VIX-IVTS**: adding the IV term-structure slope,
#    - **HAR-RV-VIX-VVIX-L-IVTS**: combining both VVIX, leverage and IV term-structure features.
#
# These HAR-RV-VIX-X variants remain parsimonious and interpretable, and serve as our main linear benchmarks against richer machine-learning models.

# %%
har_specs = { 
    "HAR-RV":                 ["RV_D", "RV_W", "RV_M"],
    "HAR-RV-VIX":             ["RV_D", "RV_W", "RV_M", "VIX"],
    "HAR-RV-VIX-VVIX":        ["RV_D", "RV_W", "RV_M", "VIX", "VVIX"],
    "HAR-RV-VIX-L":           ["RV_D", "RV_W", "RV_M", "VIX", "neg_r2"],
    "HAR-RV-VIX-IVTS":        ["RV_D", "RV_W", "RV_M", "VIX", "iv_ts"],
    "HAR-RV-VIX-VVIX-L-IVTS": ["RV_D", "RV_W", "RV_M", "VIX", "VVIX", "neg_r2", "iv_ts"]
}

dp_kwargs["scale"] = True # Need to scale

metrics_lin = []  # list of metrics dicts
y_preds_lin = {}  # name -> CV predictions (Series)

for name, feats in har_specs.items():
    metrics, y_pred = eval_model_cv(
        name=name,
        base_estimator=lin_model,
        features=feats,
        X=X_train,
        y=y_train,
        cv=purged_cv,
        dp_kwargs=dp_kwargs,
        y_pred_bench=y_naive_rv,
    )
    metrics_lin.append(metrics)
    y_preds_lin[name] = y_pred

metrics_lin = pd.DataFrame(metrics_lin).set_index("model")
display(pd.concat([metrics_bench, metrics_lin], axis=0))

# %% [markdown]
# The `HAR-RV-VIX-IVTS` specification yields the best **average** performance. However, it barely improves on `HAR-RV-VIX` despite adding one extra predictor. We therefore check whether it actually outperforms `HAR-RV-VIX` consistently across most folds.

# %%
lin_model = Pipeline([
    ("dp", DataProcessor(**dp_kwargs)),
    ("lr", LinearRegression()),
])

# HAR-RV-VIX
scores_vix = cross_val_score(
    lin_model,
    X_train[har_specs["HAR-RV-VIX"]],
    y_train,
    cv=purged_cv,
    scoring="neg_mean_squared_error",
)

# HAR-RV-VIX-IVTS
scores_vix_ivts = cross_val_score(
    lin_model,
    X_train[har_specs["HAR-RV-VIX-IVTS"]],
    y_train,
    cv=purged_cv,
    scoring="neg_mean_squared_error",
)

ph.plot_cv_mse_comparison(
    scores_vix,
    scores_vix_ivts,
    label_a="HAR-RV-VIX",
    label_b="HAR-RV-VIX-IVTS",
)

# %% [markdown]
# The `HAR-RV-VIX-IVTS` specification delivers the **highest average R²**, but only marginally so and without a clear, consistent gain across folds. In several folds it actually underperforms `HAR-RV-VIX`. Given this very small and unstable improvement for just one extra predictor, we retain `HAR-RV-VIX` as our preferred linear benchmark.

# %% [markdown]
# ## 9.2 Random Forest
#
# For the non-linear side (Random Forest), we perform **hyperparameter tuning** on the key parameters (not too many to avoid overfitting the validation set) using the same Purged K-Fold CV, and keep the RF configuration with the best validation performance.

# %%
param_grid = {
    "rf__max_depth": [3, 5, 7],
    "rf__min_samples_leaf": [5, 10, 20],
    "rf__max_features": ["sqrt", 0.5, 1.0],
}

dp_kwargs["scale"] = False # No need to scale

pipe_rf = Pipeline([
    ("dp", DataProcessor(**dp_kwargs)),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])

gscv = GridSearchCV(
    pipe_rf, param_grid=param_grid, 
    scoring="neg_mean_squared_error",
    cv=purged_cv, n_jobs=-1
)

X_train_rf = X_train[features_rf]
gscv.fit(X_train_rf, y_train)
print("Best params:", gscv.best_params_)

# %%
best_rf_pipe = gscv.best_estimator_
best_rf = best_rf_pipe.named_steps["rf"] 

metrics_rf, y_pred_rf = eval_model_cv(
    name="Random Forest",
    base_estimator=best_rf,
    features=features_rf,
    X=X_train,
    y=y_train,
    cv=purged_cv,
    dp_kwargs=dp_kwargs,
    y_pred_bench=y_naive_rv,
)
metrics_rf = pd.DataFrame([metrics_rf]).set_index("model")

display(pd.concat([metrics_bench, metrics_rf], axis=0))

# %% [markdown]
# ## 9.3 Model Comparison and Selection
#
# Here we compare the forecasts of the best linear HAR-RV-VIX model and the tuned Random Forest, and test whether simple ensembles of the two can improve overall performance and/or reduce the variance of forecast errors.

# %%
ph.plot_model_comparison_ts(
    y_train,
    y_pred_1=y_preds_lin["HAR-RV-VIX"],
    y_pred_2=y_pred_rf,
    label_1="HAR-RV-VIX",
    label_2="RF",
)

# %%
ph.plot_model_comparison_scatter(
    y_train,
    y_preds_lin["HAR-RV-VIX"], 
    y_pred_rf,
    label_1="HAR_RV_VIX",
    label_2="RF",
)

# %% [markdown]
# Here we see that **forecast errors grow when RV is high**, which is expected: extreme volatility regimes are inherently harder to predict accurately.

# %%
weights = [0.5, 0.7, 0.8]

ens_metrics, ens_preds = eval_ensembles(
    y_train,
    y_preds_lin["HAR-RV-VIX"],
    y_pred_rf,
    weights,
    y_pred_bench=y_naive_rv
)

display(pd.concat([metrics_lin.loc[["HAR-RV-VIX"]], metrics_rf, ens_metrics], axis=0))

# %% [markdown]
# ### **Conclusion**
#
# We do not find any strong evidence that an ensemble adds value here: it neither improves predictive performance nor reduces the variance of the forecast errors compared to the individual models.
#
# We therefore keep `HAR-RV-VIX` as our final linear model specification.

# %% [markdown]
# ## 9.4 Time-variation in benchmark strength (IV vs HAR-RV-VIX)

# %%
perf = pd.concat(
    [y_train.rename("y_true"),
     y_preds_lin["HAR-RV-VIX"].rename("har_vix"),
     y_naive_rv.rename("naive_rv"),
     y_naive_iv.rename("naive_iv")],
    axis=1
).dropna()

subperiods = [
    ("2010-01-01", "2012-12-31", "2010–2012"),
    ("2013-01-01", "2015-12-31", "2013–2015"),
    ("2016-01-01", "2018-12-31", "2016–2018"),
    ("2019-01-01", "2020-12-31", "2019–2020"),
]

ph.plot_subperiod_comparison(perf, subperiods)

# %%
metrics_subperiod = compute_subperiod_metrics(perf, subperiods)
display(metrics_subperiod)

# %% [markdown]
# Over time, **implied volatility becomes a stronger predictor of future realized volatility**, approaching the performance of our HAR-RV-VIX model — especially **after the Volmageddon event of February 2018**.
#
# Before 2018, the **VRP was structurally positive**, whereas **post-2018 it has often been negative**. This shift points to **distinct volatility regimes**, making **regime conditioning essential**; without it, our model will tend to underpredict periods of elevated future RV.

# %% [markdown]
#
# ---
#

# %% [markdown]
# # **10. Out-of-Sample Walk-Forward Evaluation**
#
# In this section we evaluate the **final fixed model** (`HAR-RV-VIX`) on a held-out period using a true **walk-forward** scheme.
# We compare its performance against two benchmarks:
#
# - a **Naive RV** benchmark, to measure the **forecasting edge** of our model,
# - a **HAR-RV** benchmark, to measure the added value of including VIX (an “econometric edge”).
#
# All modelling and selection decisions (feature set, model class, hyperparameters) were made on the **research sample** (`2010–2020`). The walk-forward backtest over `2021–2025` was run **once**, without ex-post tweaking of backtest parameters (rebalancing frequency, rolling-window length, etc.), in order to mitigate backtest overfitting and selection bias.

# %% [markdown]
# ## 10.1 Backtest Horizon & Data Construction
#
# We evaluate performance over the period `2021-01-01` to `2025-12-31`.
# To avoid any look-ahead, we build the feature/target panel using all
# available data from the start of the research sample up to the end of
# the backtest window.

# %%
# From start of research (2010-01-01) to end of backtest (2025-12-31)
start_backtest = "2021-01-01"
end_backtest = "2025-12-31"

es_5min_full = load_intraday_prices(
    DATA_INTER / "es-5m.csv",
    start=start,
    end=end_backtest,
)

iv_atm_30d = pd.read_csv(
    DATA_PROC / "spx_atm_iv_30d_2016_2023.csv", 
    index_col=0, parse_dates=True
)
iv_atm_30d = iv_atm_30d.loc[start_backtest:end_backtest]

X_full, y_full = rvfeat.build_har_vix_dataset(
    es_5min_full,
    h=21,
)

X_full

# %% [markdown]
# ## 10.2 Walk-Forward Configuration
#
# We use a **rolling 3-year window**, re-fitting the model **monthly**:
#
# - training window: last 3 years of data up to the rebalancing date,
# - test window: next month,
# - we **purge** the last 21 days of the training window, because the
#   21-day forward target at time *t* uses information up to *t+21*.
#
# As before, all preprocessing (log/sqrt transforms, winsorisation, scaling)
# is learned **only on the training window** and then applied to the
# corresponding test window inside the pipeline.

# %%
dp_kwargs = dict(
    log_features=log_features,
    winsor_sqrt_features=winsor_sqrt_features,
    winsor_features=winsor_features,
    scale=True
)

lin_pipe = Pipeline([
    ("dp", DataProcessor(**dp_kwargs)),
    ("lr", LinearRegression()),
])

wf = WalkForwardOOS(
    estimator=lin_pipe,
    start_backtest=start_backtest,
    end_backtest=end_backtest,
    expanding=False,   # rolling window
    window_years=3,
    rebal_freq="ME",   # month-end
    purge_horizon=21,
)

# %% [markdown]
# ## 10.3 Walk-Forward Results & Performance Metrics
#
# We now run the walk-forward procedure over the 2021–2025 backtest window and collect the **true vs predicted** 21-day log-RV for each model. On this OOS path we compare the **Naive-RV**, **HAR-RV**, and **HAR-RV-VIX** using the same metrics as in the research section (R², MSE, QLIKE, residual variance). This tells us whether the gains we saw in cross-validation actually **translate into genuine forecasting edge** on unseen data.

# %% [markdown]
# ## 10.3 Walk-Forward Results (RV Benchmarks, 2021–2025)
#
# We now run the walk-forward procedure over the 2021–2025 backtest window and
# collect the **true vs predicted** 21-day log-RV for each model. On this OOS path
# we compare **Naive-RV**, **HAR-RV**, and **HAR-RV-VIX** using the same metrics as
# in the research section (R², MSE, QLIKE, residual variance).
#
# This tells us whether the gains observed in cross-validation translate into a
# **genuine forecasting edge** on unseen data when judged purely against
# realised-volatility benchmarks.

# %%
# 1) Run WF for HAR-RV-VIX (final model) and HAR-RV (benchmark)
preds_rv_vix = wf.run(X_full, y_full)
preds_rv = wf.run(X_full.drop(columns=["VIX"]), y_full)

# no intersection with iv here – use full OOS
y_true_full     = preds_rv_vix["y_true"]
y_pred_vix_full = preds_rv_vix["y_pred"]
y_pred_har_full = preds_rv["y_pred"]

y_naive_rv_full, _ = rvfeat.build_naive_targets(
    X_full.loc[y_true_full.index, "RV_M"],
    iv_atm=None  # or just ignore IV here
)

metrics_naive_rv = compute_metrics(y_true_full, y_naive_rv_full)
metrics_har      = compute_metrics(y_true_full, y_pred_har_full,  y_pred_bench=y_naive_rv_full)
metrics_vix      = compute_metrics(y_true_full, y_pred_vix_full,  y_pred_bench=y_naive_rv_full)

metrics_naive_rv["model"] = "Naive_RV"
metrics_har["model"]      = "HAR-RV"
metrics_vix["model"]      = "HAR-RV-VIX"

# build table for 2021–2025 (Naive_RV, HAR-RV, HAR-RV-VIX)
metrics_df = pd.DataFrame(
    [metrics_naive_rv, metrics_har, metrics_vix]
).set_index("model")

display(metrics_df)

# %%
plt.figure(figsize=(12, 4))

y_true_full.plot(label="True 21D RV", lw=1.5)
y_naive_rv_full.plot(label="Naive RV", lw=1)
y_pred_har_full.plot(label="HAR-RV", lw=1)
y_pred_vix_full.plot(label="HAR-RV-VIX", lw=1.5)

plt.legend(loc="upper left")
plt.title("OOS forecasts: True vs Naive, HAR-RV, HAR-RV-VIX")
plt.ylabel("log 21D RV")
plt.tight_layout()
plt.show()

# %%
models = {
    "Naive-RV":  y_naive_rv_full,
    "HAR-RV":    y_pred_har_full,
    "HAR-RV-VIX": y_pred_vix_full,
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
xy_min = min(y_true_full.min(), *(p.min() for p in models.values()))
xy_max = max(y_true_full.max(), *(p.max() for p in models.values()))

for ax, (name, preds) in zip(axes, models.items()):
    ax.scatter(y_true_full, preds, s=8, alpha=0.6)
    ax.plot([xy_min, xy_max], [xy_min, xy_max])
    ax.set_title(name)
    ax.set_xlabel("True log 21D RV")
axes[0].set_ylabel("Predicted log 21D RV")

plt.tight_layout()
plt.show()

# %%
res_vix = y_true_full - y_pred_vix_full

plt.figure(figsize=(12, 4))
res_vix.plot(lw=0.8, label="Residuals")
res_vix.rolling(126).std().plot(lw=1.5, alpha=0.8, label="6M rolling std")
plt.axhline(0.0, color="k", lw=1)
plt.title("HAR-RV-VIX residuals over time (walk-forward OOS)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The $R^2_{\text{OOS}}$ for `HAR-RV-VIX` beats both benchmark models, providing evidence of a genuine **forecasting edge** over the naive RV benchmark and an **econometric edge** over the standard HAR-RV model.

# %% [markdown]
# ## 10.4 Walk-Forward Results vs Implied Volatility (2021–2023)
#
# Because option data (ATM 30D IV) are only available up to 2023, we repeat the
# walk-forward evaluation on the **restricted 2021–2023 window**, this time adding
# the **Naive-IV** benchmark:
#
# - **Naive-RV**: carry forward the 21-day realised variance (floor benchmark),
# - **Naive-IV**: transform ATM 30D IV into forward variance,
# - **HAR-RV**, **HAR-RV-VIX**: same fixed specifications as in the research set.
#
# This comparison answers a stronger question:  
# > *Is the model not only better than naive RV, but also more informative than
# > the volatility implied in option prices themselves?*

# %%
idx_iv = preds_rv_vix.index.intersection(iv_atm_30d.index)

y_true_iv     = preds_rv_vix.loc[idx_iv, "y_true"]
y_pred_vix_iv = preds_rv_vix.loc[idx_iv, "y_pred"]
y_pred_har_iv = preds_rv.loc[idx_iv, "y_pred"]
iv_slice      = iv_atm_30d.loc[idx_iv]

y_naive_rv_iv, y_naive_iv = rvfeat.build_naive_targets(
    X_full.loc[y_true_iv.index, "RV_M"],
    iv_slice,
)

metrics_naive_rv_iv = compute_metrics(y_true_iv, y_naive_rv_iv)
metrics_naive_iv    = compute_metrics(y_true_iv, y_naive_iv, y_pred_bench=y_naive_rv_iv)
metrics_har_iv      = compute_metrics(y_true_iv, y_pred_har_iv, y_pred_bench=y_naive_rv_iv)
metrics_vix_iv      = compute_metrics(y_true_iv, y_pred_vix_iv, y_pred_bench=y_naive_rv_iv)

metrics_naive_rv_iv["model"] = "Naive_RV"
metrics_naive_iv["model"] = "Naive-IV"
metrics_har_iv["model"] = "HAR-RV"
metrics_vix_iv["model"] = "HAR-RV-VIX"

# build table for 2021–2025 (Naive_RV, HAR-RV, HAR-RV-VIX)
metrics_df = pd.DataFrame(
    [metrics_naive_rv_iv, metrics_naive_iv, metrics_har_iv, metrics_vix_iv]
).set_index("model")

display(metrics_df)

# %%
plt.figure(figsize=(12, 4))

y_true_iv.plot(label="True 21D RV", lw=1.5)
y_naive_iv.plot(label="Naive IV", lw=1)
y_pred_vix_iv.plot(label="HAR-RV-VIX", lw=1.5)

plt.legend(loc="upper left")
plt.title("OOS forecasts: True vs Naive-IV, HAR-RV-VIX")
plt.ylabel("log 21D RV")
plt.tight_layout()
plt.show()

# %% [markdown]
# Here, the results suggest that **ATM IV** is itself a better predictor of future RV over the 2021–2023 window. In terms of **RMSE, MSE and $R^2_{\text{OOS}}$** it clearly beats our HAR-RV-VIX model, although it has slightly better **QLIKE**.
#
# However, this conclusion must be treated with caution: the backtest window is short, and because the 21-day targets overlap, we only have on the order of **30–40 effectively independent observations**. As you can see from the plot, this makes it hard to draw strong statistical conclusions from this period alone.
#
# Thus using a HAR-RV-VIX model as a trading tool to trade IV-RV msiirtpcing might be a bad idea as the market has been pricign IV better and better that for a 30 day horizon.

# %% [markdown]
# # **11. Conclusion**
#
# This study illustrates how hard it is to beat a well-specified HAR-RV benchmark. Even after adding a rich set of economically motivated predictors and testing a tuned non-linear Random Forest, we do not find convincing evidence that complex models deliver a materially better or more robust forecasting performance than a parsimonious `HAR-RV-VIX` specification.
#
# It is possible that at different horizons some predictors (e.g. very short-term or high-frequency signals) would become more useful, and that non-linear models could add more value. In our case, however, the forecast horizon was deliberately set to 21 trading days to match the 30-DTE options we intend to trade, where liquidity is highest.
#
# Under this constraint, a simple, interpretable `HAR-RV-VIX` model offers a clean baseline and a modest but robust forecasting edge, without the complexity and instability of richer machine-learning alternatives.

# %% [markdown]
# start_year = start_backtest[:4]
# end_year   = end_backtest[:4]
# y_pred_vix.index.name = "date"
#
# fname = DATA_PROC / f"har_vix_rv_predictions_{start_year}_{end_year}.csv"
# y_pred_vix.to_csv(fname)
