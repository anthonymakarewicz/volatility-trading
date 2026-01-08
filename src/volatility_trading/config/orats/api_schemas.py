"""
ORATS API schema specs (raw JSON -> intermediate parquet).

These specs are meant to support mechanical normalization:
- cast vendor fields to stable dtypes (pre-rename)
- parse vendor date/datetime fields (pre-rename)
- rename vendor -> canonical
- select canonical columns to keep in intermediate
- bounds:
    * bounds_drop_canonical: out-of-bounds => drop row (structural validity)
    * bounds_null_canonical: out-of-bounds => set to null

The API schemas are stored by logical endpoint name.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from .schema_spec import OratsSchemaSpec


# ----------------------------------------------------------------------------
# Endpoint: monies_implied
# ----------------------------------------------------------------------------

# ----- Dtypes (vendor) ------
_MONIES_IMPLIED_VENDOR_DTYPES: dict[str, pl.DataType] = {
    # Keys
    "ticker": pl.Utf8,
    "tradeDate": pl.Utf8,
    "expirDate": pl.Utf8,

    # Underlying + rates
    "stockPrice": pl.Float64,
    "spotPrice": pl.Float64,
    "riskFreeRate": pl.Float64,

    # Dividend/yield model fields
    "yieldRate": pl.Float64,
    "residualYieldRate": pl.Float64,
    "residualRateSlp": pl.Float64,
    "residualR2": pl.Float64,
    "confidence": pl.Float64,

    # ATM vols / calibration
    "atmiv": pl.Float64,
    "calVol": pl.Float64,
    "unadjVol": pl.Float64,
    "earnEffect": pl.Float64,

    # Timestamps (parse later)
    "quoteDate": pl.Utf8,
    "updatedAt": pl.Utf8,

    # Optional model diagnostics / additional fields
    "mwVol": pl.Float64,
    "typeFlag": pl.Int64,
    "slope": pl.Float64,
    "deriv": pl.Float64,
    "fit": pl.Float64,

    # Delta-slice vols
    "vol100": pl.Float64,
    "vol95": pl.Float64,
    "vol90": pl.Float64,
    "vol85": pl.Float64,
    "vol80": pl.Float64,
    "vol75": pl.Float64,
    "vol70": pl.Float64,
    "vol65": pl.Float64,
    "vol60": pl.Float64,
    "vol55": pl.Float64,
    "vol50": pl.Float64,
    "vol45": pl.Float64,
    "vol40": pl.Float64,
    "vol35": pl.Float64,
    "vol30": pl.Float64,
    "vol25": pl.Float64,
    "vol20": pl.Float64,
    "vol15": pl.Float64,
    "vol10": pl.Float64,
    "vol5": pl.Float64,
    "vol0": pl.Float64,
}

# ----- Dates / datetimes (vendor) ------
_MONIES_IMPLIED_VENDOR_DATE_COLS: tuple[str, ...] = ("tradeDate", "expirDate")
_MONIES_IMPLIED_VENDOR_DATETIME_COLS: tuple[str, ...] = ("quoteDate", "updatedAt")

# ------ Renames vendor -> canonical  ------
_MONIES_IMPLIED_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    # Keys
    "ticker": "ticker",
    "tradeDate": "trade_date",
    "expirDate": "expiry_date",

    # Underlying + rates
    "stockPrice": "underlying_price",
    "spotPrice": "spot_price",
    "riskFreeRate": "risk_free_rate",

    # Yield decomposition
    "yieldRate": "yield_rate",
    "residualYieldRate": "residual_yield_rate",
    "residualRateSlp": "residual_rate_slp",
    "residualR2": "residual_r2",
    "confidence": "confidence",

    # ATM vols / calibration
    "atmiv": "atm_iv",
    "calVol": "cal_vol",
    "unadjVol": "unadj_vol",
    "earnEffect": "earn_effect",

    # Timestamps
    "quoteDate": "quote_ts",
    "updatedAt": "updated_ts",

    # Optional diagnostics
    "mwVol": "mw_vol",
    "typeFlag": "type_flag",
    "slope": "slope",
    "deriv": "deriv",
    "fit": "fit",

    # Delta-slice vols
    "vol100": "vol_100",
    "vol95": "vol_95",
    "vol90": "vol_90",
    "vol85": "vol_85",
    "vol80": "vol_80",
    "vol75": "vol_75",
    "vol70": "vol_70",
    "vol65": "vol_65",
    "vol60": "vol_60",
    "vol55": "vol_55",
    "vol50": "vol_50",
    "vol45": "vol_45",
    "vol40": "vol_40",
    "vol35": "vol_35",
    "vol30": "vol_30",
    "vol25": "vol_25",
    "vol20": "vol_20",
    "vol15": "vol_15",
    "vol10": "vol_10",
    "vol5": "vol_5",
    "vol0": "vol_0",
}

# ------ Keep (canonical) ------
_MONIES_IMPLIED_KEEP_CANONICAL: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "expiry_date",

    "underlying_price",
    "spot_price",
    "risk_free_rate",

    "yield_rate",
    "residual_yield_rate",
    "residual_rate_slp",
    "residual_r2",
    "confidence",

    "atm_iv",
    "cal_vol",
    "unadj_vol",
    "earn_effect",

    "type_flag",
    "quote_ts",
    "updated_ts",
)

# ------ Bounds (canonical) ------

# Tier 1: structural validity => drop row
_MONIES_IMPLIED_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {
    "underlying_price": (0.0, 1e7),
    "spot_price": (0.0, 1e7),
    # expiry/trade are dates; dedupe/join logic relies on them, but that’s not a numeric bound
}

# Tier 2: value plausibility => set to null (keep row)
_MONIES_IMPLIED_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    "risk_free_rate": (-1.0, 1.0),
    "yield_rate": (-1.0, 1.0),
    "residual_yield_rate": (-1.0, 1.0),
    "residual_rate_slp": (-10.0, 10.0),

    "atm_iv": (0.0, 10.0),
    "cal_vol": (0.0, 10.0),
    "unadj_vol": (0.0, 10.0),
    "earn_effect": (-10.0, 10.0),
}


# ----------------------------------------------------------------------------
# Endpoint: summaries
# ----------------------------------------------------------------------------

# ----- Dtypes (vendor) -----
_SUMMARIES_VENDOR_DTYPES: dict[str, pl.DataType] = {
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

    "quote_ts",
    "updated_ts",
)

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


# ----------------------------------------------------------------------------
# Public mapping
# ----------------------------------------------------------------------------

API_SCHEMAS: Final[dict[str, OratsSchemaSpec]] = {
    "monies_implied": OratsSchemaSpec(
        vendor_dtypes=_MONIES_IMPLIED_VENDOR_DTYPES,
        vendor_date_cols=_MONIES_IMPLIED_VENDOR_DATE_COLS,
        vendor_datetime_cols=_MONIES_IMPLIED_VENDOR_DATETIME_COLS,
        renames_vendor_to_canonical=_MONIES_IMPLIED_RENAMES_VENDOR_TO_CANONICAL,
        keep_canonical=_MONIES_IMPLIED_KEEP_CANONICAL,
        bounds_drop_canonical=_MONIES_IMPLIED_BOUNDS_DROP_CANONICAL,
        bounds_null_canonical=_MONIES_IMPLIED_BOUNDS_NULL_CANONICAL,
    ),
    "summaries": OratsSchemaSpec(
        vendor_dtypes=_SUMMARIES_VENDOR_DTYPES,
        vendor_date_cols=_SUMMARIES_VENDOR_DATE_COLS,
        vendor_datetime_cols=_SUMMARIES_VENDOR_DATETIME_COLS,
        renames_vendor_to_canonical=_SUMMARIES_RENAMES_VENDOR_TO_CANONICAL,
        keep_canonical=_SUMMARIES_KEEP_CANONICAL,
        bounds_drop_canonical=_SUMMARIES_BOUNDS_DROP_CANONICAL,
        bounds_null_canonical=_SUMMARIES_BOUNDS_NULL_CANONICAL,
    ),
}


def get_schema_spec(endpoint: str) -> OratsSchemaSpec:
    """Return the schema spec for a supported ORATS API endpoint.

    Raises
    ------
    KeyError
        If the endpoint is unknown.
    """
    try:
        return API_SCHEMAS[endpoint]
    except KeyError as e:
        supported = ", ".join(sorted(API_SCHEMAS.keys()))
        raise KeyError(
            f"Unknown ORATS API schema for endpoint '{endpoint}'. "
            f"Supported: {supported}"
        ) from e