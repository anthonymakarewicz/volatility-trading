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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **ORATS SPY Options Chain EDA**
#
# This notebook keeps the original EDA intent while sourcing QC diagnostics from
# the pipeline artifact `qc_summary.json` instead of re-implementing manual checks
# in notebook cells.

# %%
# %load_ext autoreload
# %autoreload 2

import json
from datetime import date

import pandas as pd
import polars as pl
import yfinance as yf

from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_daily_features,
    scan_options_chain,
)

try:
    from notebooks.qc_eda.helpers import QCSummaryHelper
    from notebooks.qc_eda.plotting import (
        plot_avg_volume_by_delta,
        plot_greeks_vs_strike,
        plot_iv_time_series_with_slope,
        plot_liquidity_by_dte,
        plot_term_structure_samples,
        plot_smiles_by_delta,
        plot_spot_vs_yahoo,
        plot_term_structures_by_delta,
    )
except ModuleNotFoundError:
    from helpers import QCSummaryHelper
    from plotting import (
        plot_avg_volume_by_delta,
        plot_greeks_vs_strike,
        plot_iv_time_series_with_slope,
        plot_liquidity_by_dte,
        plot_term_structure_samples,
        plot_smiles_by_delta,
        plot_spot_vs_yahoo,
        plot_term_structures_by_delta,
    )


# %% [markdown]
# # Read SPY Options data
#
# We analyze the whole chain from `2007-01-01` to `2025-12-31` and keep contracts
# inside a broad tradable region.

# %%
TICKER = "SPY"

start = date(2007, 1, 1)
end = date(2025, 12, 5)

delta_min = 0.01
delta_max = 0.99
dte_min = 5
dte_max = 252

lf = scan_options_chain(TICKER)
lf = lf.filter(
    pl.col("trade_date").is_between(start, end),
    pl.col("call_delta").abs().is_between(delta_min, delta_max),
    pl.col("put_delta").abs().is_between(delta_min, delta_max),
    pl.col("dte").is_between(dte_min, dte_max),
)

df = lf.collect()
df_long = options_chain_wide_to_long(df).collect()

df

# %%
daily_features = read_daily_features(TICKER)
daily_features = daily_features.filter(
    pl.col("trade_date").is_between(start, end)
)
daily_features = daily_features.to_pandas().set_index("trade_date")
daily_features

# %% [markdown]
# # Load QC summary artifact
#
# The checks below are read from the quality checks summary after running `orats-api-download --config config/orats_api_download.yml`
# ```

# %%
qc_summary_path = (
     PROC_ORATS_OPTIONS_CHAIN / f"underlying={TICKER}" / "qc_summary.json"
)

with qc_summary_path.open(encoding="utf-8") as f:
    qc_summary = json.load(f)

qc_helpers = QCSummaryHelper(qc_summary)

len(qc_summary), qc_summary[3]

# %% [markdown]
# # **GLOBAL vs ROI Interpretation Policy**
#
# We use this interpretation policy throughout the notebook for every QC family.
#
# - **GLOBAL**: the full options universe after broad filters.
# - **ROI**: our practical trading region of interest, roughly:
#   - moneyness around **10-90 delta**,
#   - maturity around **10-60 DTE**.
#
# Why this split matters:
#
# - Violations in far wings or extreme maturities can be real but less relevant
#   to tradable workflows.
# - Persistent violations inside ROI are more likely to affect execution quality,
#   strategy risk sizing, and signal reliability.
#
# Severity policy used in sections below:
#
# 1. **HARD** checks: structural invalid data (drop-candidate rows).
# 2. **SOFT** checks: investigate rate + location first (GLOBAL vs ROI).
# 3. **INFO** diagnostics: descriptive metrics, not pass/fail by themselves.

# %% [markdown]
# ## Liquidity Context for ROI vs GLOBAL
#
# We visualise liquidity before detailed QC interpretation so ROI-vs-GLOBAL
# weighting is grounded in observable market depth.

# %% [markdown]
# ## 1) Liquidity by moneyness: volume by $\Delta$

# %%
plot_avg_volume_by_delta(df_long)

# %% [markdown]
# Deep OTM puts are usually much more traded than symmetric OTM calls, largely
# reflecting structural hedging demand.

# %% [markdown]
# ## 2) Liquidity by maturity: volume/open interest by DTE

# %%
plot_liquidity_by_dte(df_long)

# %% [markdown]
# Short maturities often carry higher traded volume relative to open interest,
# while longer maturities tend to accumulate open interest with lower turnover.

# %% [markdown]
# # **Basic Checks**
#
# Hard structural checks + calendar-level dataset checks from the QC summary.

# %%
df.describe(percentiles=(0.25, 0.5, 0.75, 0.9))

# %% [markdown]
# ## 1) Hard key integrity checks (non-negotiable)
#
# Policy:
# - Keys required to identify one contract observation should never be null.
# - Any material violation is a data-integrity blocker.

# %%
df.with_columns(qT = pl.col("yte") * pl.col("dividend_yield")).filter(pl.col("qT") > 0.05)

# %%
basic_checks = qc_helpers.qc_table(
    [
        "keys_not_null",
    ]
)
basic_checks

# %% [markdown]
# ## 2) Calendar-level dataset checks (GLOBAL diagnostics)
#
# Policy:
# - Missing exchange sessions and non-trading dates are dataset-level diagnostics.
# - Small, explainable exceptions are investigated before escalation.

# %%
basic_checks = qc_helpers.qc_table(
    [
        "keys_not_null",
        "trade_date_leq_expiry_date",
        "GLOBAL_missing_sessions_xnys",
        "GLOBAL_non_trading_dates_present_xnys",
    ]
)
basic_checks

# %%
missing_sessions = qc_helpers.qc_details("GLOBAL_missing_sessions_xnys").get("missing_dates", [])
non_trading = qc_helpers.qc_details("GLOBAL_non_trading_dates_present_xnys").get("extra_dates", [])

print("Missing XNYS sessions:", len(missing_sessions))
print("Non-trading dates present:", len(non_trading))
print("Sample missing sessions:", missing_sessions[:5])
print("Sample non-trading dates:", non_trading[:5])

# %% [markdown]
# After investgiation the trading date `2018-12-05` coreesponds to George W bush memorial day where the NYSE was closed but here
# the options market were still opened so we can assume that we can trade on this day.

# %%
df.filter(pl.col("trade_date") == pl.date(2018, 12, 5))

# %% [markdown]
# As you can see contracts were traded that day.

# %% [markdown]
# # **Days-to-expiry check**
#
#
# Here a `HARD` error will be that the current `trade_date` is larger than the exipiry
# whihc is imposisble as at the latest the trade_date can match the maturity on the expiry date but beyind it is impossible.
#
# We alos check the distributuon of the `dte` column expecting dte ranging from the tradebale filters we have applied.

# %% [markdown]
# ## 1) Hard trade-date vs expiry consistency

# %%
dte_checks = qc_helpers.qc_table(["trade_date_leq_expiry_date"])
dte_checks

# %% [markdown]
# ## 2) DTE distribution sanity (INFO diagnostics)

# %%
global_dte_stats = qc_helpers.info_stats_metric("GLOBAL_core_numeric_stats", "dte")
print("GLOBAL DTE stats")
display(global_dte_stats)

# %% [markdown]
# # **Quote sanity checks**
#
# Here we separate quote checks into 3 groups:
#
# 1. **Hard data errors** (drop-candidate rows)
#    - `negative_quotes`: bid or ask below zero (impossible market quotes)
#    - `crossed_market`: bid above ask (invalid quote state)
#    - `bid_ask_sane`: hard guardrail summary for bid/ask consistency
#
# 2. **Suspicious but often explainable microstructure cases** (investigate first)
#    - `locked_market`: bid equals ask
#    - `one_sided_quotes`: no bid with positive ask
#    These can happen, especially in low-liquidity wings or near close.
#
# 3. **Spread quality diagnostics**
#    - `wide_spread` and `very_wide_spread`
#    Not always a data error, but a trading-quality warning.
#    In practice, we care whether they cluster outside the tradable ROI.

# %% [markdown]
# ## 1) Hard quote errors (non-negotiable)
#
# Policy:
# - If these appear materially, treat affected rows as invalid candidates.
# - Expected outcome in clean data: near-zero violation rates.

# %%
hard_quote_checks = qc_helpers.qc_table(
    [
        "bid_ask_sane",
        "negative_quotes",
        "crossed_market",
    ]
)
hard_quote_checks

# %% [markdown]
# ## 2) Locked and one-sided quotes (investigate, then decide)
#
# Policy:
# - Keep as soft flags first.
# - Escalate only if rates are high in ROI (10-60 DTE, 10-90 delta).

# %%
microstructure_quote_checks = qc_helpers.qc_table(
    [
        "GLOBAL_locked_market_C",
        "GLOBAL_locked_market_P",
        "ROI_locked_market_C",
        "ROI_locked_market_P",
        "GLOBAL_one_sided_quotes_C",
        "GLOBAL_one_sided_quotes_P",
        "ROI_one_sided_quotes_C",
        "ROI_one_sided_quotes_P",
    ]
)
microstructure_quote_checks

# %%
qc_helpers.qc_top_buckets("GLOBAL_one_sided_quotes_P")

# %%
qc_helpers.qc_top_buckets("ROI_one_sided_quotes_P")

# %% [markdown]
# ## 3) Spread diagnostics (execution quality)
#
# Policy:
# - Wide spreads are expected to be much worse in wings / short DTE.
# - We focus on ROI behavior to assess strategy impact.

# %%
spread_quote_checks = qc_helpers.qc_table(
    [
        "GLOBAL_wide_spread_C",
        "GLOBAL_wide_spread_P",
        "ROI_wide_spread_C",
        "ROI_wide_spread_P",
        "GLOBAL_very_wide_spread_C",
        "GLOBAL_very_wide_spread_P",
        "ROI_very_wide_spread_C",
        "ROI_very_wide_spread_P",
    ]
)
spread_quote_checks

# %%
qc_helpers.qc_top_buckets("GLOBAL_wide_spread_P")

# %%
qc_helpers.qc_top_buckets("ROI_wide_spread_P")

# %% [markdown]
# # **Volume & Open Interest Checks**
#
# Here we separate volume/OI checks into 3 groups:
#
# 1. **Hard data errors** (drop-candidate rows)
#    - `negative_vol_oi`: negative traded volume or open interest (invalid values)
#
# 2. **Soft consistency diagnostics** (investigate first)
#    - `zero_vol_pos_oi`: positive OI with zero volume
#    - `pos_vol_zero_oi`: positive volume with zero OI
#    These are often explainable by microstructure/timing but can cluster in weak
#    quality regions.
#
# 3. **INFO liquidity diagnostics**
#    - `volume_oi_metrics` summaries for GLOBAL and ROI scopes
#    Not a pass/fail rule; used to profile tradability and market depth.

# %% [markdown]
# ## 1) Hard volume/OI sign errors (non-negotiable)

# %%
hard_vol_oi_checks = qc_helpers.qc_table(["negative_vol_oi"])
hard_vol_oi_checks

# %% [markdown]
# ## 2) Soft volume/OI mismatch diagnostics

# %%
soft_vol_oi_checks = qc_helpers.qc_table(
    [
        "GLOBAL_zero_vol_pos_oi_C",
        "GLOBAL_zero_vol_pos_oi_P",
        "ROI_zero_vol_pos_oi_C",
        "ROI_zero_vol_pos_oi_P",
        "GLOBAL_pos_vol_zero_oi_C",
        "GLOBAL_pos_vol_zero_oi_P",
        "ROI_pos_vol_zero_oi_C",
        "ROI_pos_vol_zero_oi_P",
    ]
)
soft_vol_oi_checks

# %% [markdown]
# ## 3) INFO liquidity metrics (GLOBAL vs ROI)

# %%
vol_oi_metrics = pd.DataFrame(
    [
        qc_helpers.qc_details("GLOBAL_volume_oi_metrics"),
        qc_helpers.qc_details("ROI_volume_oi_metrics"),
    ],
    index=["GLOBAL_volume_oi_metrics", "ROI_volume_oi_metrics"],
)
vol_oi_metrics

# %%
qc_helpers.qc_top_buckets("GLOBAL_zero_vol_pos_oi_P")

# %%
qc_helpers.qc_top_buckets("GLOBAL_pos_vol_zero_oi_P")

# %% [markdown]
# # **Spot price sanity checks**
#
# Spot consistency checks come from the QC summary; external price comparison
# remains as EDA context.

# %% [markdown]
# ## 1) Structural spot consistency checks (QC summary)
#
# Here we chekc that the spot price is the same for a given (ticker, trade_date, expiry_date, strike) bucket, namely that it is the same accros teh whole chain for a given day.
#
# In the case of ETF/Stock options, there is no implied froward price whihc is store din the `underlying_price` column so the `spot_price` shoudl be the same as `underlying_price`.

# %%
spot_checks = qc_helpers.qc_table(
    [
        "GLOBAL_spot_constant_per_trade_date",
        "GLOBAL_spot_equals_underlying_per_trade_date",
    ]
)
spot_checks

# %% [markdown]
# ## 2) ORATS SPY vs Yahoo Finance non-adjusted close
#
# In ORATS, the options chain data provies a complete snapshot of the US equity options market 14 minutes before the close of trading each day.
#
# Thus we should expect it to be very close to another data source, for instance Yahoo finance data which is reliable for a highly liquid ETF liek `SPY`.

# %%
spx_yf = yf.download(TICKER, start=start, end=end, auto_adjust=False)["Close"]
spx_yf = spx_yf.squeeze()
spx_yf.name = "spy_yf_close"

spx_orats = (
    df.group_by("trade_date")
    .agg(pl.col("spot_price").first().alias("spy_orats_spot"))
    .sort("trade_date")
    .to_pandas()
    .set_index("trade_date")
)

spx = pd.concat([spx_yf, spx_orats], axis=1).dropna()

diff = spx["spy_orats_spot"] - spx["spy_yf_close"]
rel_diff = diff / spx["spy_yf_close"]
corr = spx.corr().loc["spy_yf_close", "spy_orats_spot"]

print("Correlation (ORATS spot vs Yahoo close):", corr)
display(
    pd.DataFrame({"abs_diff": diff.abs(), "rel_diff": rel_diff}).describe(
        percentiles=[0.5, 0.9, 0.99]
    )
)

# %%
plot_spot_vs_yahoo(spx)

# %% [markdown]
# The two are very close to this confirms that the spot price data form ORATS is of good quality.

# %% [markdown]
# # **Dividend Yield checks**
#
# Dividend yield enters several model-based diagnostics (notably parity bounds
# through the carry term `qT`). We first inspect summary stats, then inspect the
# cross-DTE shape on sample days.

# %% [markdown]
# ## 1) INFO summary statistics (GLOBAL and ROI)

# %%
qc_helpers.info_stats_metric("GLOBAL_core_numeric_stats", "dividend_yield")

# %%
qc_helpers.info_stats_metric("ROI_core_numeric_stats", "dividend_yield")

# %% [markdown]
# Typical levels (mean/median near ~2%) are plausible for SPY, while the upper
# tail is large in unconditional stats. The term-structure view below is used to
# separate structural ex-dividend effects from true outlier/noise behavior.

# %% [markdown]
# ## 2) Term-structure diagnostics with ex-dividend anchors
#
# We overlay Yahoo Finance ex-dividend dates as dashed vertical markers in DTE
# space for each sample day. This helps check whether sharp jumps in the
# `dividend_yield` curve align with upcoming ex-dividend events.

# %%
sample_days = [
    date(2007, 1, 3),
    date(2012, 6, 15),
    date(2018, 12, 24),
    date(2025, 1, 3),
]

spy_dividends = yf.Ticker(TICKER).dividends
ex_div_dates = sorted(
    {
        pd.Timestamp(ts).date()
        for ts in spy_dividends.index
        if start <= pd.Timestamp(ts).date() <= end
    }
)

plot_term_structure_samples(
    df,
    sample_days=sample_days,
    value_col="dividend_yield",
    ex_div_dates=ex_div_dates,
)

# %% [markdown]
# In these sample days, the main **dividend-yield jumps** align with **ex-dividend
# anchors**, which supports a **structural carry** interpretation rather than random
# data corruption.

# %% [markdown]
# # **Risk free rate check**

# %% [markdown]
# ## 1) Structural uniqueness check (per day-expiry bucket)

# %%
rf_checks = qc_helpers.qc_table(
    [
        "GLOBAL_unique_risk_free_rate_per_day_expiry",
    ]
)
rf_checks

# %% [markdown]
# ## 2) INFO risk-free metrics (GLOBAL vs ROI)

# %%
pd.DataFrame(
    [
        qc_helpers.qc_details("GLOBAL_risk_free_rate_metrics"),
        qc_helpers.qc_details("ROI_risk_free_rate_metrics"),
    ],
    index=["GLOBAL_risk_free_rate_metrics", "ROI_risk_free_rate_metrics"],
)

# %% [markdown]
# ## 3) Term-structure sanity on sample days

# %%
sample_days = [
    date(2007, 1, 3),
    date(2012, 6, 15),
    date(2018, 12, 24),
    date(2025, 1, 3),
]

plot_term_structure_samples(df, sample_days=sample_days, value_col="risk_free_rate")

# %% [markdown]
# Here ORATS is using a 4-point yield curve per day, one for short maturities (less than 30 DTE) and one for int  (30 <= 90) and one for (90< dte <180) aned the last one beyond 180.

# %% [markdown]
# # **Implied Volatility Quality Checks**
#
# We split IV diagnostics into one HARD validity check and two SOFT tail checks.
#
# - `iv_non_negative` is a **HARD** check: implied volatility should never be
#   negative in clean data.
#
# - `GLOBAL_high_iv` uses a **100% IV threshold** (`smoothed_iv > 1.0`).
#   This can occur during extreme stress regimes (for example, crisis windows).
#
# - `GLOBAL_very_high_iv` uses a **200% IV threshold** (`smoothed_iv > 2.0`).
#   This is much stricter and should be rare.
#

# %% [markdown]
# ## 1) Hard and soft IV threshold checks

# %%
iv_checks = qc_helpers.qc_table(
    [
        "iv_non_negative",
        "GLOBAL_high_iv",
        "GLOBAL_very_high_iv",
    ]
)
iv_checks

# %% [markdown]
# ## 2) INFO IV distribution diagnostics

# %%
qc_helpers.info_stats_metric("GLOBAL_core_numeric_stats", "smoothed_iv")

# %% [markdown]
# ## 3) Smile Shapes
#
# We use **delta** as the moneyness measure so that calls and puts can be placed on a
# single, continuous implied-volatility curve.
#
# ORATS provides a **smoothed implied-volatility surface** (SMV) that is shared across
# calls and puts. This lets us analyse the smile consistently across option types and
# maturities (e.g., 10, 30, and 60 DTE).

# %%
picked_dates = [
    date(2008, 10, 10),
    date(2010, 12, 2),
    date(2013, 6, 13),
    date(2015, 8, 24),  
    date(2018, 2, 5),  
    date(2018, 9, 12),
    date(2020, 3, 16), 
    date(2022, 6, 16), 
    date(2025, 3, 3),
]

event_labels = {
    date(2008, 10, 10): "GFC stress",
    date(2015, 8, 24): "China / flash crash",
    date(2018, 2, 5): "Volmageddon",
    date(2020, 3, 16): "Covid crash",
    date(2022, 6, 16): "Rates/Inflation stress",
}

plot_smiles_by_delta(df, picked_dates=picked_dates, event_labels=event_labels)

# %% [markdown]
# Because the **IV values we plot are already SMV-smoothed**, we do not apply any
# additional smoothing. A simple interpolation across reported points is sufficient
# when we want to visualise a continuous curve. In practice, if we need IV at a
# specific target (e.g., a strike or delta bucket), we either:
# - **select the closest available quote** (nearest neighbour), or
# - **linearly interpolate** between adjacent quotes (if the target lies between them).
#
# For details on the ORATS smoothing methodology (SMV system), see:
# https://orats.com/blog/smoothing-options-implied-volatilities-using-orats-smv-system

# %% [markdown]
# ## 4) Term-Structure Shapes
#
# This figure plots **implied-volatility term structures** (Smoothed IV vs **DTE**) for a few **delta buckets**
# across multiple **trade dates** (one facet per date).

# %%
plot_term_structures_by_delta(df, picked_dates=picked_dates, event_labels=event_labels)

# %% [markdown]
# On **crash / stress dates**, the term structure often becomes **inverted (backwardation)**: short-dated IV rises
# relative to longer-dated IV. In more **normal regimes**, it is typically **upward sloping (contango)**, with
# longer-dated IV above short-dated IV.

# %% [markdown]
# ## 5) Implied Volatility Time-Series
#
# The figure shows the evolution of smoothed implied volatility over time for
# multiple maturities (e.g., 10D, 30D, 90D, 1Y).
#
# Each series reflects the market’s forward-looking risk expectations over a
# different horizon:
# - short-dated IV (10–30D) captures **near-term uncertainty**
# - medium maturities (90D) reflect **quarterly risk**
# - long-dated IV (1Y) embeds **structural / long-run expectations**

# %%
plot_iv_time_series_with_slope(daily_features, event_labels=event_labels)

# %% [markdown]
# Short-dated IV is more reactive than longer maturities, especially during crises,
# as markets price immediate uncertainty over the next weeks.
#
# Longer maturities (e.g., 1Y) also rise but more smoothly since they embed
# expectations over a full year, including the anticipated post-shock recovery.
#
# The lower panel (`10D - 1Y`) is a compact slope diagnostic:
# - positive values indicate short-dated stress dominance (backwardation),
# - negative values indicate a more normal contango-like term shape.

# %% [markdown]
# # **Greeks Sanity Checks**
#
# We split Greeks QC into HARD data errors and SOFT diagnostics.
#
# 1. **HARD sign errors (drop-candidate rows)**
#    - `gamma_non_negative`
#    - `vega_non_negative`
#    These are treated as structural issues in a clean options chain.
#    In the pipeline they are HARD checks with a tiny numeric tolerance:
#    violation if `gamma < -1e-8` or `vega < -1e-8`.
#
# 2. **SOFT diagnostics (investigate rate and location first)**
#    - `*_delta_bounds_sane_*` for calls/puts, GLOBAL and ROI
#    - `*_theta_positive_*` for calls/puts, GLOBAL and ROI
#
#    Delta theoretical bounds are:
#    $$
#    0 \le \Delta_C \le 1,\qquad -1 \le \Delta_P \le 0.
#    $$
#    We still allow small numeric noise at row level (`eps=1e-5`), then judge
#    the **violation rate** with soft thresholds.
#
#    Positive theta is also SOFT (row tolerance `eps=1e-8`) because it can be
#    legitimate in some cases, for example:
#    - dividend/carry effects (especially American options),
#    - deep ITM contracts with early-exercise effects,
#    - very short-maturity edge cases and vendor-Greeks approximation noise.

# %% [markdown]
# ## 1) QC summary table (hard + soft checks)

# %%
greeks_checks_cols = [
    "gamma_non_negative",
    "vega_non_negative",
    "GLOBAL_delta_bounds_sane_C",
    "GLOBAL_delta_bounds_sane_P",
    "ROI_delta_bounds_sane_C",
    "ROI_delta_bounds_sane_P",
    "GLOBAL_theta_positive_C",
    "GLOBAL_theta_positive_P",
    "ROI_theta_positive_C",
    "ROI_theta_positive_P",
]

qc_helpers.qc_table(greeks_checks_cols)

# %% [markdown]
# ## 2) Soft-threshold policy view

# %%
qc_helpers.qc_thresholds(greeks_checks_cols)

# %% [markdown]
# ## 3) Greeks vs Strike
#
# Here we investigate how option Greeks vary with strike for both calls and puts,
# highlighting the typical theoretical shapes observed around the ATM region and
# in the wings.

# %%
day = date(2024, 12, 16)
dte_target = 30

plot_greeks_vs_strike(df_long, day=day, dte_target=dte_target)

# %% [markdown]
# - **Delta** should be monotonic in strike: from near `+1` (deep ITM calls) toward
#   `0` (deep OTM calls). For puts, delta typically lies in `[-1, 0]` under the
#   standard sign convention.
# - **Gamma** and **Vega** usually peak near ATM and decay in the wings.
# - **Theta** is typically negative for long options near ATM, though localized
#   positive-theta pockets can appear (carry/dividend effects and, for American
#   options, early-exercise features).
#
# **Note on ORATS conventions:** ORATS uses a consistent convention for quoting
# Greeks across calls and puts (e.g., shared formulations and sign conventions).
# When comparing call/put surfaces, interpret values according to this convention.
# See: [ORATS – “Option Greeks are the same for calls and puts”](https://orats.com/blog/option-greeks-are-the-same-for-calls-and-puts).

# %% [markdown]
# # **Model-driven / arbitrage checks**
#
# This section follows the GLOBAL-vs-ROI interpretation policy defined above.
# We interpret arbitrage diagnostics with higher weight on ROI behavior.

# %% [markdown]
# ## 1) Put-call parity diagnostics
#
# **Economic context (AOA):**
# In frictionless markets, no-arbitrage implies a strict parity relation for
# European options:
#
# $$
# C_E - P_E = S_0 e^{-qT} - K e^{-rT}.
# $$
#
# **Tradable arbitrage context:**
# In live markets, we cannot trade at mid and we pay bid/ask costs. So a small
# parity gap is often non-actionable. We therefore assess parity with a
# spread-aware tolerance rather than as an exact equality.

# %% [markdown]
# ### American parity check used in this QC
#
# SPY options are American, so we use bounds (not equality):
#
# $$
# S_0 e^{-qT} - K \le C_A - P_A \le S_0 - K e^{-rT}.
# $$
#
# Using mid prices:
#
# $$
# L = C_{\text{mid}} - P_{\text{mid}}.
# $$
#
# Dynamic tolerance:
#
# $$
# \tau = \alpha\Big((C_{\text{ask}}-C_{\text{bid}}) + (P_{\text{ask}}-P_{\text{bid}})\Big) + \tau_0.
# $$
#
# In this pipeline, $\alpha = 1.0$ and $\tau_0 = 0.01$.
# We flag a violation when:
#
# $$
# L < \text{lower} - \tau \quad \text{or} \quad L > \text{upper} + \tau.
# $$
#
# This is a SOFT data-quality diagnostic: high violations in ROI are more
# concerning than violations concentrated in illiquid wings.

# %%
pcp_checks_cols = [
    "GLOBAL_pcp_bounds_mid_am",
    "ROI_pcp_bounds_mid_am",
]

qc_helpers.qc_table(pcp_checks_cols)

# %%
qc_helpers.qc_top_buckets(pcp_checks_cols[0])

# %%
qc_helpers.qc_top_buckets(pcp_checks_cols[1])

# %% [markdown]
# Here a large propotion of the violations happen in the 0.3 0.7 delta bucket 
#
# Investgtae dividend yield col

# %% [markdown]
# ## 2) Price-bounds diagnostics (calls and puts)
#
# We monitor two no-arbitrage envelopes and treat violations as SOFT diagnostics.
#
# ### American-spot bounds
#
# Spot-based bounds used as an additional diagnostic:
#
# $$
# \max(0, S_0-K) \le C_{\text{mid}} \le S_0,
# $$
# $$
# \max(0, K-S_0) \le P_{\text{mid}} \le K.
# $$
#
#
# Bounds are evaluated with spread-aware tolerance:
#
# $$
# \tau=\max\!\big(\tau_0,\alpha\,|\text{ask}-\text{bid}|\big),
# $$
#
# with $\alpha=1.0$ and $\tau_0=0.01$ in this pipeline.
# A row is flagged when:
#
# $$
# \text{mid} < \text{lower} - \tau \quad \text{or} \quad \text{mid} > \text{upper} + \tau.
# $$
#
# Interpretation: small rates outside ROI can be microstructure noise; persistent
# ROI violations are more concerning for tradable strategies.

# %%
bounds_checks = qc_helpers.qc_table(
    [
        "GLOBAL_price_bounds_mid_am_C",
        "GLOBAL_price_bounds_mid_am_P",
        "ROI_price_bounds_mid_am_C",
        "ROI_price_bounds_mid_am_P",
    ]
)
bounds_checks

# %% [markdown]
# A **large number of price-bound violations** is observed across calls, puts, and the tradeable ROI. To localise the issue, we analyse breaches using **DTE × delta buckets**.

# %%
qc_helpers.qc_top_buckets("ROI_price_bounds_mid_am_C")

# %%
qc_helpers.qc_top_buckets("ROI_price_bounds_mid_am_P")

# %% [markdown]
# - Violations are **heavily concentrated in low-delta wings**, particularly in the **(0.1, 0.3] bucket** across short and medium maturities, where **rates exceed ~65%**.
#
# - By contrast, the **(0.3, 0.7] region (near-ATM, more tradeable)** shows **much lower violation rates** and a **smaller share of total breaches**.
#
# Overall, **global violation metrics overstate the practical impact**: most inconsistencies arise in **far-OTM, lower-liquidity regions**, not in the **core tradable surface**.

# %% [markdown]
# ## 3) Monotonicity diagnostics
#
# We treat monotonicity checks as SOFT diagnostics and separate them into strike-
# based and maturity-based arbitrage interpretations.

# %% [markdown]
# ### 2.1 Vertical spread arbitrage (strike monotonicity)
#
# For fixed trade date and expiry, strike monotonicity conditions are:
#
# $$
# C(K_1, T) \ge C(K_2, T) \quad \text{for } K_1 < K_2,
# $$
# $$
# P(K_1, T) \le P(K_2, T) \quad \text{for } K_1 < K_2.
# $$
#
# In practice (American exercise + quote noise), we treat violations as SOFT and
# judge impact using GLOBAL vs ROI concentration.

# %%
strike_monotonicity_checks = qc_helpers.qc_table(
    [
        "GLOBAL_strike_monotonicity_C",
        "GLOBAL_strike_monotonicity_P",
        "ROI_strike_monotonicity_C",
        "ROI_strike_monotonicity_P",
    ]
)
strike_monotonicity_checks

# %% [markdown]
# The violation rate is low (less than 2%) across all the checks except the `GLOBAL_maturity_monotonicity_P`. Thus we inspect the location of those violations usign the **Delta x Dte buckets**

# %%
qc_helpers.qc_top_buckets("GLOBAL_strike_monotonicity_P")

# %% [markdown]
# **More than 50%** of the violations observed before are located in the very etxreme wings so we cna safely ignore them.

# %% [markdown]
# ### 2.3 Calendar arbitrage (maturity monotonicity)
#
# At fixed strike, maturity monotonicity is:
#
# $$
# C(K, T_2) \ge C(K, T_1), \qquad P(K, T_2) \ge P(K, T_1)
# \quad \text{for } T_2 > T_1.
# $$
#
# We monitor maturity-order violations as SOFT diagnostics, with emphasis on ROI
# where tradability impact is highest.

# %%
maturity_monotonicity_checks = qc_helpers.qc_table(
    [
        "GLOBAL_maturity_monotonicity_C",
        "GLOBAL_maturity_monotonicity_P",
        "ROI_maturity_monotonicity_C",
        "ROI_maturity_monotonicity_P",
    ]
)
maturity_monotonicity_checks

# %%
qc_helpers.qc_top_buckets("GLOBAL_maturity_monotonicity_P").head(10)
