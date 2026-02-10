"""ORATS API schema spec for endpoint `summaries`.

Defines vendor-to-canonical normalization for summary-level market context
fields such as term-structure IVs, rates, borrow, and dividend estimates.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from ..schema_spec import OratsSchemaSpec

# ----- Dtypes (vendor) -----
_SUMMARIES_VENDOR_DTYPES: dict[str, type[pl.DataType]] = {
    "ticker": pl.Utf8,
    "tradeDate": pl.Utf8,
    "stockPrice": pl.Float64,
    "annActDiv": pl.Float64,
    "annIdiv": pl.Float64,
    "nextDiv": pl.Float64,
    "impliedNextDiv": pl.Float64,
    "borrow30": pl.Float64,
    "borrow2y": pl.Float64,
    "riskFree30": pl.Float64,
    "riskFree2y": pl.Float64,
    "confidence": pl.Float64,
    "totalErrorConf": pl.Float64,
    "iv10d": pl.Float64,
    "iv20d": pl.Float64,
    "iv30d": pl.Float64,
    "iv60d": pl.Float64,
    "iv90d": pl.Float64,
    "iv6m": pl.Float64,
    "iv1y": pl.Float64,
    # Delta-slice IVs (still useful; keep in intermediate if you want later)
    "dlt5Iv10d": pl.Float64,
    "dlt5Iv20d": pl.Float64,
    "dlt5Iv30d": pl.Float64,
    "dlt5Iv60d": pl.Float64,
    "dlt5Iv90d": pl.Float64,
    "dlt5Iv6m": pl.Float64,
    "dlt5Iv1y": pl.Float64,
    "dlt25Iv10d": pl.Float64,
    "dlt25Iv20d": pl.Float64,
    "dlt25Iv30d": pl.Float64,
    "dlt25Iv60d": pl.Float64,
    "dlt25Iv90d": pl.Float64,
    "dlt25Iv6m": pl.Float64,
    "dlt25Iv1y": pl.Float64,
    "dlt75Iv10d": pl.Float64,
    "dlt75Iv20d": pl.Float64,
    "dlt75Iv30d": pl.Float64,
    "dlt75Iv60d": pl.Float64,
    "dlt75Iv90d": pl.Float64,
    "dlt75Iv6m": pl.Float64,
    "dlt75Iv1y": pl.Float64,
    "dlt95Iv10d": pl.Float64,
    "dlt95Iv20d": pl.Float64,
    "dlt95Iv30d": pl.Float64,
    "dlt95Iv60d": pl.Float64,
    "dlt95Iv90d": pl.Float64,
    "dlt95Iv6m": pl.Float64,
    "dlt95Iv1y": pl.Float64,
    # Timestamps (parse later)
    "quoteDate": pl.Utf8,
    "updatedAt": pl.Utf8,
}

# ----- Dates / datetimes (vendor) ------
_SUMMARIES_VENDOR_DATE_COLS: tuple[str, ...] = ("tradeDate",)
_SUMMARIES_VENDOR_DATETIME_COLS: tuple[str, ...] = ("quoteDate", "updatedAt")

# ------ Renames vendor -> canonical ------
_SUMMARIES_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    "ticker": "ticker",
    "tradeDate": "trade_date",
    "stockPrice": "underlying_price",
    "annActDiv": "ann_actual_div",
    "annIdiv": "ann_implied_div",
    "nextDiv": "next_div",
    "impliedNextDiv": "implied_next_div",
    "borrow30": "borrow_rate_30d",
    "borrow2y": "borrow_rate_2y",
    "riskFree30": "risk_free_rate_30d",
    "riskFree2y": "risk_free_rate_2y",
    "confidence": "confidence",
    "totalErrorConf": "total_error_conf",
    "iv10d": "iv_10d",
    "iv20d": "iv_20d",
    "iv30d": "iv_30d",
    "iv60d": "iv_60d",
    "iv90d": "iv_90d",
    "iv6m": "iv_6m",
    "iv1y": "iv_1y",
    # Delta-slice IVs
    "dlt5Iv10d": "iv_dlt05_10d",
    "dlt5Iv20d": "iv_dlt05_20d",
    "dlt5Iv30d": "iv_dlt05_30d",
    "dlt5Iv60d": "iv_dlt05_60d",
    "dlt5Iv90d": "iv_dlt05_90d",
    "dlt5Iv6m": "iv_dlt05_6m",
    "dlt5Iv1y": "iv_dlt05_1y",
    "dlt25Iv10d": "iv_dlt25_10d",
    "dlt25Iv20d": "iv_dlt25_20d",
    "dlt25Iv30d": "iv_dlt25_30d",
    "dlt25Iv60d": "iv_dlt25_60d",
    "dlt25Iv90d": "iv_dlt25_90d",
    "dlt25Iv6m": "iv_dlt25_6m",
    "dlt25Iv1y": "iv_dlt25_1y",
    "dlt75Iv10d": "iv_dlt75_10d",
    "dlt75Iv20d": "iv_dlt75_20d",
    "dlt75Iv30d": "iv_dlt75_30d",
    "dlt75Iv60d": "iv_dlt75_60d",
    "dlt75Iv90d": "iv_dlt75_90d",
    "dlt75Iv6m": "iv_dlt75_6m",
    "dlt75Iv1y": "iv_dlt75_1y",
    "dlt95Iv10d": "iv_dlt95_10d",
    "dlt95Iv20d": "iv_dlt95_20d",
    "dlt95Iv30d": "iv_dlt95_30d",
    "dlt95Iv60d": "iv_dlt95_60d",
    "dlt95Iv90d": "iv_dlt95_90d",
    "dlt95Iv6m": "iv_dlt95_6m",
    "dlt95Iv1y": "iv_dlt95_1y",
    "quoteDate": "quote_ts",
    "updatedAt": "updated_ts",
}


# ------ Bounds (canonical) ------
_SUMMARIES_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {
    "underlying_price": (0.0, 1e7),
}

_SUMMARIES_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    # Term structure IV
    "iv_10d": (0.0, 10.0),
    "iv_20d": (0.0, 10.0),
    "iv_30d": (0.0, 10.0),
    "iv_60d": (0.0, 10.0),
    "iv_90d": (0.0, 10.0),
    "iv_6m": (0.0, 10.0),
    "iv_1y": (0.0, 10.0),
    # Delta-slice IVs
    "iv_dlt25_10d": (0.0, 10.0),
    "iv_dlt25_20d": (0.0, 10.0),
    "iv_dlt25_30d": (0.0, 10.0),
    "iv_dlt25_60d": (0.0, 10.0),
    "iv_dlt25_90d": (0.0, 10.0),
    "iv_dlt25_6m": (0.0, 10.0),
    "iv_dlt25_1y": (0.0, 10.0),
    "iv_dlt75_10d": (0.0, 10.0),
    "iv_dlt75_20d": (0.0, 10.0),
    "iv_dlt75_30d": (0.0, 10.0),
    "iv_dlt75_60d": (0.0, 10.0),
    "iv_dlt75_90d": (0.0, 10.0),
    "iv_dlt75_6m": (0.0, 10.0),
    "iv_dlt75_1y": (0.0, 10.0),
    # Rates/borrow
    "risk_free_rate_30d": (-1.0, 1.0),
    "risk_free_rate_2y": (-1.0, 1.0),
    "borrow_rate_30d": (-1.0, 1.0),
    "borrow_rate_2y": (-1.0, 1.0),
    # Divs (wide but finite)
    "ann_actual_div": (-1e6, 1e6),
    "ann_implied_div": (-1e6, 1e6),
    "next_div": (-1e6, 1e6),
    "implied_next_div": (-1e6, 1e6),
}


# ------ Keep (canonical) ------
_SUMMARIES_KEEP_CANONICAL: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "underlying_price",
    "ann_actual_div",
    "ann_implied_div",
    "next_div",
    "implied_next_div",
    "borrow_rate_30d",
    "borrow_rate_2y",
    "risk_free_rate_30d",
    "risk_free_rate_2y",
    "confidence",
    "total_error_conf",
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
    "quote_ts",
    "updated_ts",
)


# ----------------------------------------------------------------------------
# Public schema spec
# ----------------------------------------------------------------------------

SUMMARIES_SCHEMA: Final[OratsSchemaSpec] = OratsSchemaSpec(
    vendor_dtypes=_SUMMARIES_VENDOR_DTYPES,
    vendor_date_cols=_SUMMARIES_VENDOR_DATE_COLS,
    vendor_datetime_cols=_SUMMARIES_VENDOR_DATETIME_COLS,
    renames_vendor_to_canonical=_SUMMARIES_RENAMES_VENDOR_TO_CANONICAL,
    keep_canonical=_SUMMARIES_KEEP_CANONICAL,
    bounds_drop_canonical=_SUMMARIES_BOUNDS_DROP_CANONICAL,
    bounds_null_canonical=_SUMMARIES_BOUNDS_NULL_CANONICAL,
)
