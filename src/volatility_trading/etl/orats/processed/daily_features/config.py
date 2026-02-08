"""
Daily features schema for processed ORATS panels.

A daily-features panel is keyed by (ticker, trade_date) and contains
low-frequency predictors sourced from ORATS API endpoints.
"""

from __future__ import annotations

DAILY_FEATURES_CORE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "iv_10d",
    "iv_20d",
    "iv_30d",
    "iv_60d",
    "iv_90d",
    "iv_6m",
    "iv_1y",
    "iv_dlt25_10d",
    "iv_dlt25_20d",
    "iv_dlt25_30d",
    "iv_dlt25_60d",
    "iv_dlt25_90d",
    "iv_dlt25_6m",
    "iv_dlt25_1y",
    "iv_dlt75_10d",
    "iv_dlt75_20d",
    "iv_dlt75_30d",
    "iv_dlt75_60d",
    "iv_dlt75_90d",
    "iv_dlt75_6m",
    "iv_dlt75_1y",
    "hv_intra_1d",
    "hv_intra_5d",
    "hv_intra_10d",
    "hv_intra_20d",
    "hv_intra_30d",
    "hv_intra_60d",
    "hv_intra_90d",
    "hv_intra_100d",
    "hv_intra_120d",
    "hv_intra_252d",
)


# ----------------------------------------------------------------------------
# Endpoint -> minimal columns needed for processed daily_features
# ----------------------------------------------------------------------------

DAILY_FEATURES_ENDPOINT_COLUMNS: dict[str, tuple[str, ...]] = {
    "summaries": (
        "ticker",
        "trade_date",
        "iv_10d",
        "iv_20d",
        "iv_30d",
        "iv_60d",
        "iv_90d",
        "iv_6m",
        "iv_1y",
        "iv_dlt25_10d",
        "iv_dlt25_20d",
        "iv_dlt25_30d",
        "iv_dlt25_60d",
        "iv_dlt25_90d",
        "iv_dlt25_6m",
        "iv_dlt25_1y",
        "iv_dlt75_10d",
        "iv_dlt75_20d",
        "iv_dlt75_30d",
        "iv_dlt75_60d",
        "iv_dlt75_90d",
        "iv_dlt75_6m",
        "iv_dlt75_1y",
    ),
    "hvs": (
        "ticker",
        "trade_date",
        "hv_intra_1d",
        "hv_intra_5d",
        "hv_intra_10d",
        "hv_intra_20d",
        "hv_intra_30d",
        "hv_intra_60d",
        "hv_intra_90d",
        "hv_intra_100d",
        "hv_intra_120d",
        "hv_intra_252d",
    ),
}


# Exact per-column multipliers (endpoint -> {col_name -> multiplier})
DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS: dict[str, dict[str, float]] = {
    # Example (uncomment if needed later):
    # "summaries": {
    #     "risk_free_rate_30d": 0.01,  # 5.0 -> 0.05
    #     "borrow_rate_30d": 0.01,
    # },
    "hvs": {
        # keep empty if you only use glob for hvs
    },
}

# Glob/pattern multipliers (endpoint -> [(glob_pattern, multiplier), ...])
# Uses fnmatch-style globs: hv_intra_* , risk_free_rate_* , etc.
DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS_GLOB: dict[str, list[tuple[str, float]]] = {
    "hvs": [
        ("hv_intra_*", 0.01),  # ORATS HVS is like 12.5 meaning 12.5% -> 0.125
    ],
}

# If True: raise if a column matches multiple glob patterns in the same endpoint.
# (Protects you from accidental double-scaling.)
DAILY_FEATURES_UNITS_STRICT: bool = True
