"""
ORATS API schema spec for endpoint: hvs (historical volatility).

This endpoint provides:
- historical intraday volatility ("orHv*")
- historical close-to-close volatility ("clsHv*")
- variants excluding day-of and day-after earnings ("*Xern*")
"""

from __future__ import annotations

from typing import Final

import polars as pl

from ..schema_spec import OratsSchemaSpec


# ----------------------------------------------------------------------------
# Vendor dtypes
# ----------------------------------------------------------------------------

_HVS_VENDOR_DTYPES: dict[str, pl.DataType] = {
    "ticker": pl.Utf8,
    "tradeDate": pl.Utf8,

    # Intraday historical vol (ORATS: orHv*)
    "orHv1d": pl.Float64,
    "orHv5d": pl.Float64,
    "orHv10d": pl.Float64,
    "orHv20d": pl.Float64,
    "orHv30d": pl.Float64,
    "orHv60d": pl.Float64,
    "orHv90d": pl.Float64,
    "orHv100d": pl.Float64,
    "orHv120d": pl.Float64,
    "orHv252d": pl.Float64,
    "orHv500d": pl.Float64,
    "orHv1000d": pl.Float64,

    # Close-to-close historical vol (ORATS: clsHv*)
    "clsHv5d": pl.Float64,
    "clsHv10d": pl.Float64,
    "clsHv20d": pl.Float64,
    "clsHv30d": pl.Float64,
    "clsHv60d": pl.Float64,
    "clsHv90d": pl.Float64,
    "clsHv100d": pl.Float64,
    "clsHv120d": pl.Float64,
    "clsHv252d": pl.Float64,
    "clsHv500d": pl.Float64,
    "clsHv1000d": pl.Float64,

    # Intraday vol excluding day-of and day-after earnings (ORATS: orHvXern*)
    "orHvXern5d": pl.Float64,
    "orHvXern10d": pl.Float64,
    "orHvXern20d": pl.Float64,
    "orHvXern30d": pl.Float64,
    "orHvXern60d": pl.Float64,
    "orHvXern90d": pl.Float64,
    "orHvXern100d": pl.Float64,
    "orHvXern120d": pl.Float64,
    "orHvXern252d": pl.Float64,
    "orHvXern500d": pl.Float64,
    "orHvXern1000d": pl.Float64,

    # Close-to-close vol excluding day-of and day-after earnings (clsHvXern*)
    "clsHvXern5d": pl.Float64,
    "clsHvXern10d": pl.Float64,
    "clsHvXern20d": pl.Float64,
    "clsHvXern30d": pl.Float64,
    "clsHvXern60d": pl.Float64,
    "clsHvXern90d": pl.Float64,
    "clsHvXern100d": pl.Float64,
    "clsHvXern120d": pl.Float64,
    "clsHvXern252d": pl.Float64,
    "clsHvXern500d": pl.Float64,
    "clsHvXern1000d": pl.Float64,

    # Common ORATS timestamp field on many endpoints
    "updatedAt": pl.Utf8,
}


# ----------------------------------------------------------------------------
# Vendor date / datetime cols
# ----------------------------------------------------------------------------

_HVS_VENDOR_DATE_COLS: tuple[str, ...] = ("tradeDate",)
_HVS_VENDOR_DATETIME_COLS: tuple[str, ...] = ("updatedAt",)


# ----------------------------------------------------------------------------
# Renames vendor -> canonical (Option A)
# ----------------------------------------------------------------------------

_HVS_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    "ticker": "ticker",
    "tradeDate": "trade_date",

    # Intraday historical vol
    "orHv1d": "hv_intra_1d",
    "orHv5d": "hv_intra_5d",
    "orHv10d": "hv_intra_10d",
    "orHv20d": "hv_intra_20d",
    "orHv30d": "hv_intra_30d",
    "orHv60d": "hv_intra_60d",
    "orHv90d": "hv_intra_90d",
    "orHv100d": "hv_intra_100d",
    "orHv120d": "hv_intra_120d",
    "orHv252d": "hv_intra_252d",
    "orHv500d": "hv_intra_500d",
    "orHv1000d": "hv_intra_1000d",

    # Close-to-close historical vol
    "clsHv5d": "hv_close_5d",
    "clsHv10d": "hv_close_10d",
    "clsHv20d": "hv_close_20d",
    "clsHv30d": "hv_close_30d",
    "clsHv60d": "hv_close_60d",
    "clsHv90d": "hv_close_90d",
    "clsHv100d": "hv_close_100d",
    "clsHv120d": "hv_close_120d",
    "clsHv252d": "hv_close_252d",
    "clsHv500d": "hv_close_500d",
    "clsHv1000d": "hv_close_1000d",

    # Intraday ex-earnings
    "orHvXern5d": "hv_intra_xern_5d",
    "orHvXern10d": "hv_intra_xern_10d",
    "orHvXern20d": "hv_intra_xern_20d",
    "orHvXern30d": "hv_intra_xern_30d",
    "orHvXern60d": "hv_intra_xern_60d",
    "orHvXern90d": "hv_intra_xern_90d",
    "orHvXern100d": "hv_intra_xern_100d",
    "orHvXern120d": "hv_intra_xern_120d",
    "orHvXern252d": "hv_intra_xern_252d",
    "orHvXern500d": "hv_intra_xern_500d",
    "orHvXern1000d": "hv_intra_xern_1000d",

    # Close-to-close ex-earnings
    "clsHvXern5d": "hv_close_xern_5d",
    "clsHvXern10d": "hv_close_xern_10d",
    "clsHvXern20d": "hv_close_xern_20d",
    "clsHvXern30d": "hv_close_xern_30d",
    "clsHvXern60d": "hv_close_xern_60d",
    "clsHvXern90d": "hv_close_xern_90d",
    "clsHvXern100d": "hv_close_xern_100d",
    "clsHvXern120d": "hv_close_xern_120d",
    "clsHvXern252d": "hv_close_xern_252d",
    "clsHvXern500d": "hv_close_xern_500d",
    "clsHvXern1000d": "hv_close_xern_1000d",

    "updatedAt": "updated_ts",
}


# ----------------------------------------------------------------------------
# Bounds (canonical)
# ----------------------------------------------------------------------------

_HVS_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {}

_HVS_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    # HV values are vol (not %); keep wide but finite.
    # If ORATS returns percentages, your normalization layer should convert.
    "hv_intra_1d": (0.0, 10.0),
    "hv_intra_5d": (0.0, 10.0),
    "hv_intra_10d": (0.0, 10.0),
    "hv_intra_20d": (0.0, 10.0),
    "hv_intra_30d": (0.0, 10.0),
    "hv_intra_60d": (0.0, 10.0),
    "hv_intra_90d": (0.0, 10.0),
    "hv_intra_100d": (0.0, 10.0),
    "hv_intra_120d": (0.0, 10.0),
    "hv_intra_252d": (0.0, 10.0),
    "hv_intra_500d": (0.0, 10.0),
    "hv_intra_1000d": (0.0, 10.0),

    "hv_close_5d": (0.0, 10.0),
    "hv_close_10d": (0.0, 10.0),
    "hv_close_20d": (0.0, 10.0),
    "hv_close_30d": (0.0, 10.0),
    "hv_close_60d": (0.0, 10.0),
    "hv_close_90d": (0.0, 10.0),
    "hv_close_100d": (0.0, 10.0),
    "hv_close_120d": (0.0, 10.0),
    "hv_close_252d": (0.0, 10.0),
    "hv_close_500d": (0.0, 10.0),
    "hv_close_1000d": (0.0, 10.0),

    "hv_intra_xern_5d": (0.0, 10.0),
    "hv_intra_xern_10d": (0.0, 10.0),
    "hv_intra_xern_20d": (0.0, 10.0),
    "hv_intra_xern_30d": (0.0, 10.0),
    "hv_intra_xern_60d": (0.0, 10.0),
    "hv_intra_xern_90d": (0.0, 10.0),
    "hv_intra_xern_100d": (0.0, 10.0),
    "hv_intra_xern_120d": (0.0, 10.0),
    "hv_intra_xern_252d": (0.0, 10.0),
    "hv_intra_xern_500d": (0.0, 10.0),
    "hv_intra_xern_1000d": (0.0, 10.0),

    "hv_close_xern_5d": (0.0, 10.0),
    "hv_close_xern_10d": (0.0, 10.0),
    "hv_close_xern_20d": (0.0, 10.0),
    "hv_close_xern_30d": (0.0, 10.0),
    "hv_close_xern_60d": (0.0, 10.0),
    "hv_close_xern_90d": (0.0, 10.0),
    "hv_close_xern_100d": (0.0, 10.0),
    "hv_close_xern_120d": (0.0, 10.0),
    "hv_close_xern_252d": (0.0, 10.0),
    "hv_close_xern_500d": (0.0, 10.0),
    "hv_close_xern_1000d": (0.0, 10.0),
}


# ----------------------------------------------------------------------------
# Keep (canonical)
# ----------------------------------------------------------------------------

_HVS_KEEP_CANONICAL: tuple[str, ...] = (
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
    "hv_intra_500d",
    "hv_intra_1000d",

    "hv_intra_xern_5d",
    "hv_intra_xern_10d",
    "hv_intra_xern_20d",
    "hv_intra_xern_30d",
    "hv_intra_xern_60d",
    "hv_intra_xern_90d",
    "hv_intra_xern_100d",
    "hv_intra_xern_120d",
    "hv_intra_xern_252d",
    "hv_intra_xern_500d",
    "hv_intra_xern_1000d",

    "updated_ts",
)


# ----------------------------------------------------------------------------
# Public schema spec
# ----------------------------------------------------------------------------

HVS_SCHEMA_SPEC: Final[OratsSchemaSpec] = OratsSchemaSpec(
    vendor_dtypes=_HVS_VENDOR_DTYPES,
    vendor_date_cols=_HVS_VENDOR_DATE_COLS,
    vendor_datetime_cols=_HVS_VENDOR_DATETIME_COLS,
    renames_vendor_to_canonical=_HVS_RENAMES_VENDOR_TO_CANONICAL,
    keep_canonical=_HVS_KEEP_CANONICAL,
    bounds_drop_canonical=_HVS_BOUNDS_DROP_CANONICAL,
    bounds_null_canonical=_HVS_BOUNDS_NULL_CANONICAL,
)