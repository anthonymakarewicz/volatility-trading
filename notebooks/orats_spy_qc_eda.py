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
# # **ORATS SPY Options Chain EDA**
#
# This notebook keeps the original EDA intent while sourcing QC diagnostics from
# the pipeline artifact `qc_summary.json` instead of re-implementing manual checks
# in notebook cells.

# %%
# %load_ext autoreload
# %autoreload 2

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import yfinance as yf

from volatility_trading.datasets import options_chain_wide_to_long, scan_options_chain
from volatility_trading.etl.orats.qc.plotting import (
    plot_avg_volume_by_delta,
    plot_liquidity_by_dte,
    plot_smiles_by_delta,
    plot_term_structures_by_delta,
)
from volatility_trading.iv_surface.term_structure import pick_closest_dte

# %% [markdown]
# # Read SPY Options data
#
# We analyze the whole chain from `2007-01-01` to `2025-12-31` and keep contracts
# inside a broad tradable region.

# %%
TICKER = "SPY"

start = date(2007, 1, 1)
end = date(2025, 12, 31)

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

# %% [markdown]
# # Load QC summary artifact
#
# The checks below are read from the quality checks summary after running `orats-api-download --config config/orats_api_download.yml`
# ```

# %%
from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN

qc_summary_path = (
     PROC_ORATS_OPTIONS_CHAIN / f"underlying={TICKER}" / "qc_summary.json"
)

with qc_summary_path.open(encoding="utf-8") as f:
    qc_summary = json.load(f)

qc_by_name = {row["name"]: row for row in qc_summary}


def qc_table(names: list[str]) -> pl.DataFrame:
    """Return a compact QC table for check names that exist."""
    rows: list[dict[str, object]] = []
    for name in names:
        row = qc_by_name.get(name)
        if row is None:
            continue
        rows.append(
            {
                "name": row["name"],
                "severity": row["severity"],
                "grade": row["grade"],
                "passed": row["passed"],
                "n_rows": row.get("n_rows"),
                "n_units": row.get("n_units"),
                "n_viol": row.get("n_viol"),
                "viol_rate": row.get("viol_rate"),
            }
        )
    return pl.DataFrame(rows).sort(["severity", "name"])


def qc_top_buckets(name: str) -> pl.DataFrame:
    """Return top bucket diagnostics for one SOFT check."""
    row = qc_by_name.get(name, {})
    top_buckets = row.get("details", {}).get("top_buckets", [])
    return pl.DataFrame(top_buckets)


def qc_details(name: str) -> dict:
    """Return details payload for one QC check name."""
    return qc_by_name.get(name, {}).get("details", {})


def first_existing(*candidates: str) -> str | None:
    """Return first check name found in qc_summary."""
    for c in candidates:
        if c in qc_by_name:
            return c
    return None


def info_stats_metric(info_name: str, metric: str) -> pd.DataFrame:
    """Return one metric block from INFO core_numeric_stats."""
    stats = qc_details(info_name).get("stats", {})
    if metric not in stats:
        return pd.DataFrame()
    out = pd.DataFrame([stats[metric]])
    out.insert(0, "metric", metric)
    return out


len(qc_summary), qc_summary[0]

# %% [markdown]
# # **Basic Checks**
#
# Hard structural checks + calendar-level dataset checks from the QC summary.

# %%
basic_checks = qc_table(
    [
        "keys_not_null",
    ]
)
basic_checks

# %%
basic_checks = qc_table(
    [
        "keys_not_null",
        "trade_date_leq_expiry_date",
        "GLOBAL_missing_sessions_xnys",
        "GLOBAL_non_trading_dates_present_xnys",
    ]
)
basic_checks

# %%
missing_sessions = qc_details("GLOBAL_missing_sessions_xnys").get("missing_dates", [])
non_trading = qc_details("GLOBAL_non_trading_dates_present_xnys").get("extra_dates", [])

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

# %%
dte_checks = qc_table(["trade_date_leq_expiry_date"])
dte_checks

# %%
global_dte_stats = info_stats_metric("GLOBAL_core_numeric_stats", "dte")
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
hard_quote_checks = qc_table(
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
microstructure_quote_checks = qc_table(
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
qc_top_buckets("GLOBAL_one_sided_quotes_P").head(10)

# %%
qc_top_buckets("ROI_one_sided_quotes_P").head(10)

# %% [markdown]
# ## 3) Spread diagnostics (execution quality)
#
# Policy:
# - Wide spreads are expected to be much worse in wings / short DTE.
# - We focus on ROI behavior to assess strategy impact.

# %%
spread_quote_checks = qc_table(
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
qc_top_buckets("GLOBAL_wide_spread_P").head(10)

# %%
qc_top_buckets("ROI_wide_spread_P").head(10)

# %% [markdown]
# # **Volume & Open Interest Checks**
#
# Hard sign checks + soft mismatch diagnostics + INFO volume/OI metrics.

# %%
vol_oi_checks = qc_table(
    [
        "negative_vol_oi",
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
vol_oi_checks

# %%
pd.DataFrame(
    [
        qc_details("GLOBAL_volume_oi_metrics"),
        qc_details("ROI_volume_oi_metrics"),
    ],
    index=["GLOBAL_volume_oi_metrics", "ROI_volume_oi_metrics"],
)

# %%
qc_top_buckets("GLOBAL_pos_vol_zero_oi_P").head(10)

# %% [markdown]
# # **Spot price sanity checks**
#
# Spot consistency checks come from the QC summary; external price comparison
# remains as EDA context.

# %%
spot_checks = qc_table(
    [
        "GLOBAL_spot_constant_per_trade_date",
        "GLOBAL_spot_equals_underlying_per_trade_date",
    ]
)
spot_checks

# %% [markdown]
# ## ORATS SPY vs Yahoo Finance Non-adjusted Closing Price

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
spx.plot(figsize=(12, 6))
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("ORATS Spot vs Yahoo SPY Close")
plt.show()

# %% [markdown]
# # **Parity-Implied Forward Index Price Check**
#
# Surface-level parity consistency checks are sourced from QC summary keys.

# %%
forward_or_spot_dataset_check = first_existing(
    "GLOBAL_forward_constant_per_trade_date_expiry",  # EU naming
    "GLOBAL_spot_equals_underlying_per_trade_date",  # AM naming
)

if forward_or_spot_dataset_check is None:
    print("No exercise-style-specific forward/spot dataset check found.")
else:
    qc_table([forward_or_spot_dataset_check])

# %% [markdown]
# # **Risk free rate check**

# %%
rf_checks = qc_table(
    [
        "GLOBAL_unique_risk_free_rate_per_day_expiry",
    ]
)
rf_checks

# %%
pd.DataFrame(
    [
        qc_details("GLOBAL_risk_free_rate_metrics"),
        qc_details("ROI_risk_free_rate_metrics"),
    ],
    index=["GLOBAL_risk_free_rate_metrics", "ROI_risk_free_rate_metrics"],
)

# %% [markdown]
# # **Implied Volatility Quality Checks**

# %%
iv_checks = qc_table(
    [
        "iv_non_negative",
        "GLOBAL_high_iv",
        "GLOBAL_very_high_iv",
    ]
)
iv_checks

# %%
display(info_stats_metric("GLOBAL_core_numeric_stats", "smoothed_iv"))
display(info_stats_metric("ROI_core_numeric_stats", "smoothed_iv"))

# %% [markdown]
# ## Smile shape for 10, 30 and 60 DTE

# %%
picked_dates = [
    date(2008, 10, 10),  # GFC
    date(2010, 12, 2),
    date(2013, 6, 13),
    date(2015, 8, 24),  # vol event
    date(2018, 2, 5),  # volmageddon
    date(2018, 9, 12),
    date(2020, 3, 16),  # covid crash
    date(2022, 6, 16),  # high vol / rates
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
# ## IV Term-Structure Shapes

# %%
plot_term_structures_by_delta(df, picked_dates=picked_dates, event_labels=event_labels)

# %% [markdown]
# # **Greeks Sanity Checks**
#
# Use hard + soft QC results and keep a strike slice plot for intuition.

# %%
greeks_checks = qc_table(
    [
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
)
greeks_checks

# %%
qc_top_buckets("GLOBAL_theta_positive_P").head(10)

# %% [markdown]
# ## Greeks vs Strike

# %%
day = date(2024, 12, 16)
dte_target = 30

sub = df_long.filter(pl.col("trade_date") == day)
dtes_for_day = sub.select(pl.col("dte").unique()).sort("dte").to_series().to_list()

dte_true = pick_closest_dte(dtes_for_day, dte_target, max_tol=10)
if dte_true is None:
    raise ValueError(f"No DTE within 10 days of target={dte_target} on {day}")

sub = sub.filter(pl.col("dte") == dte_true).sort("strike")
S = sub.select("underlying_price").to_series().item(0)
calls = sub.filter(pl.col("option_type") == "C")
puts = sub.filter(pl.col("option_type") == "P")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
ax_d, ax_g = axes[0]
ax_v, ax_t = axes[1]

plots = [
    ("delta", ax_d, "Delta", "Delta vs strike"),
    ("gamma", ax_g, "Gamma", "Gamma vs strike"),
    ("vega", ax_v, "Vega", "Vega vs strike"),
    ("theta", ax_t, "Theta", "Theta vs strike"),
]

for col, ax, ylabel, title in plots:
    ax.plot(calls["strike"], calls[col], label=f"Call {col}", marker="o")
    ax.plot(puts["strike"], puts[col], label=f"Put {col}", marker="o")
    ax.axvline(S, linestyle="--", linewidth=0.8, label="Spot S")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

ax_v.set_xlabel("Strike")
ax_t.set_xlabel("Strike")
fig.suptitle(f"Greeks vs strike - {day}, DTE={dte_true}, S={S:.2f}")
fig.tight_layout()
plt.show()

# %% [markdown]
# # **Model-driven / arbitrage checks**
#
# Liquidity context remains EDA-driven.

# %% [markdown]
# ## Volume for Calls/Puts by $\Delta$ Moneyness

# %%
plot_avg_volume_by_delta(df_long)

# %% [markdown]
# ## Volume/Open Interest by DTE

# %%
plot_liquidity_by_dte(df_long)

# %% [markdown]
# # **Put-Call Parity checks**
#
# Pull parity diagnostics from summary keys (AM/EU aware).

# %%
pcp_checks = qc_table(
    [
        "GLOBAL_pcp_mid_eu_forward",
        "ROI_pcp_mid_eu_forward",
        "GLOBAL_pcp_bounds_mid_am",
        "ROI_pcp_bounds_mid_am",
    ]
)
pcp_checks

# %%
pcp_global_name = first_existing("GLOBAL_pcp_mid_eu_forward", "GLOBAL_pcp_bounds_mid_am")
if pcp_global_name is not None:
    qc_top_buckets(pcp_global_name).head(10)

# %% [markdown]
# # **Arbitrage bounds for call & put prices**
#
# Price-bound diagnostics from summary keys.

# %%
bounds_checks = qc_table(
    [
        "GLOBAL_price_bounds_mid_eu_forward_C",
        "GLOBAL_price_bounds_mid_eu_forward_P",
        "ROI_price_bounds_mid_eu_forward_C",
        "ROI_price_bounds_mid_eu_forward_P",
        "GLOBAL_price_bounds_mid_am_C",
        "GLOBAL_price_bounds_mid_am_P",
        "ROI_price_bounds_mid_am_C",
        "ROI_price_bounds_mid_am_P",
    ]
)
bounds_checks

# %%
bounds_global_put = first_existing(
    "GLOBAL_price_bounds_mid_eu_forward_P",
    "GLOBAL_price_bounds_mid_am_P",
)
if bounds_global_put is not None:
    qc_top_buckets(bounds_global_put).head(10)

# %% [markdown]
# ## Strike & maturity monotonicity checks
#
# Use SOFT monotonicity checks from qc summary.

# %%
monotonicity_checks = qc_table(
    [
        "GLOBAL_strike_monotonicity_C",
        "GLOBAL_strike_monotonicity_P",
        "ROI_strike_monotonicity_C",
        "ROI_strike_monotonicity_P",
        "GLOBAL_maturity_monotonicity_C",
        "GLOBAL_maturity_monotonicity_P",
        "ROI_maturity_monotonicity_C",
        "ROI_maturity_monotonicity_P",
    ]
)
monotonicity_checks

# %%
qc_top_buckets("GLOBAL_strike_monotonicity_P").head(10)

# %%
qc_top_buckets("GLOBAL_maturity_monotonicity_P").head(10)
