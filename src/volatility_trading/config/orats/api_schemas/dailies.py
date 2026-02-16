"""ORATS API schema spec for endpoint `dailies`.

Defines vendor-to-canonical normalization for daily underlying OHLCV fields,
including adjusted and unadjusted price/volume series.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from ..schema_spec import OratsSchemaSpec

# ----------------------------------------------------------------------------
# Dtypes (vendor)
# ----------------------------------------------------------------------------

_DAILIES_VENDOR_DTYPES: dict[str, type[pl.DataType]] = {
    "ticker": pl.Utf8,
    "tradeDate": pl.Utf8,
    "clsPx": pl.Float64,
    "hiPx": pl.Float64,
    "loPx": pl.Float64,
    "open": pl.Float64,
    "stockVolume": pl.Int64,
    "unadjClsPx": pl.Float64,
    "unadjHiPx": pl.Float64,
    "unadjLoPx": pl.Float64,
    "unadjOpen": pl.Float64,
    "unadjStockVolume": pl.Int64,
    "updatedAt": pl.Utf8,
}


# ----------------------------------------------------------------------------
# Vendor date / datetime cols
# ----------------------------------------------------------------------------

_DAILIES_VENDOR_DATE_COLS: tuple[str, ...] = ("tradeDate",)
_DAILIES_VENDOR_DATETIME_COLS: tuple[str, ...] = ("updatedAt",)


# ----------------------------------------------------------------------------
# Renames vendor -> canonical
# ----------------------------------------------------------------------------

_DAILIES_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    "ticker": "ticker",
    "tradeDate": "trade_date",
    "clsPx": "adjusted_close_price",
    "hiPx": "adjusted_high_price",
    "loPx": "adjusted_low_price",
    "open": "adjusted_open_price",
    "stockVolume": "adjusted_volume",
    "unadjClsPx": "unadjusted_close_price",
    "unadjHiPx": "unadjusted_high_price",
    "unadjLoPx": "unadjusted_low_price",
    "unadjOpen": "unadjusted_open_price",
    "unadjStockVolume": "unadjusted_volume",
    "updatedAt": "updated_ts",
}


# ----------------------------------------------------------------------------
# Bounds (canonical)
# ----------------------------------------------------------------------------

_DAILIES_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {}

_DAILIES_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    "adjusted_open_price": (0.0, 1e7),
    "adjusted_high_price": (0.0, 1e7),
    "adjusted_low_price": (0.0, 1e7),
    "adjusted_close_price": (0.0, 1e7),
    "adjusted_volume": (0.0, 1e12),
    "unadjusted_open_price": (0.0, 1e7),
    "unadjusted_high_price": (0.0, 1e7),
    "unadjusted_low_price": (0.0, 1e7),
    "unadjusted_close_price": (0.0, 1e7),
    "unadjusted_volume": (0.0, 1e12),
}


# ----------------------------------------------------------------------------
# Keep (canonical) - keep all endpoint fields for now
# ----------------------------------------------------------------------------

_DAILIES_KEEP_CANONICAL: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "adjusted_open_price",
    "adjusted_high_price",
    "adjusted_low_price",
    "adjusted_close_price",
    "adjusted_volume",
    "unadjusted_open_price",
    "unadjusted_high_price",
    "unadjusted_low_price",
    "unadjusted_close_price",
    "unadjusted_volume",
    "updated_ts",
)


# ----------------------------------------------------------------------------
# Public schema spec
# ----------------------------------------------------------------------------

DAILIES_SCHEMA: Final[OratsSchemaSpec] = OratsSchemaSpec(
    vendor_dtypes=_DAILIES_VENDOR_DTYPES,
    vendor_date_cols=_DAILIES_VENDOR_DATE_COLS,
    vendor_datetime_cols=_DAILIES_VENDOR_DATETIME_COLS,
    renames_vendor_to_canonical=_DAILIES_RENAMES_VENDOR_TO_CANONICAL,
    keep_canonical=_DAILIES_KEEP_CANONICAL,
    bounds_drop_canonical=_DAILIES_BOUNDS_DROP_CANONICAL,
    bounds_null_canonical=_DAILIES_BOUNDS_NULL_CANONICAL,
)
