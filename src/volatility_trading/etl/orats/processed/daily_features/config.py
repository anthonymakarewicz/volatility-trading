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