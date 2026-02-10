"""ORATS API schema spec for endpoint `monies_implied`.

Defines vendor-to-canonical normalization for term-structure level implied data
including rates, yields, ATM vols, and supporting calibration diagnostics.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from ..schema_spec import OratsSchemaSpec

# ----- Dtypes (vendor) ------
_MONIES_IMPLIED_VENDOR_DTYPES: dict[str, type[pl.DataType]] = {
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


# ------ Bounds (canonical) ------

# Tier 1: structural validity => drop row
_MONIES_IMPLIED_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {
    "underlying_price": (0.0, 1e7),
    "spot_price": (0.0, 1e7),
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


# ----------------------------------------------------------------------------
# Public schema spec
# ----------------------------------------------------------------------------

MONIES_IMPLIED_SCHEMA: Final[OratsSchemaSpec] = OratsSchemaSpec(
    vendor_dtypes=_MONIES_IMPLIED_VENDOR_DTYPES,
    vendor_date_cols=_MONIES_IMPLIED_VENDOR_DATE_COLS,
    vendor_datetime_cols=_MONIES_IMPLIED_VENDOR_DATETIME_COLS,
    renames_vendor_to_canonical=_MONIES_IMPLIED_RENAMES_VENDOR_TO_CANONICAL,
    keep_canonical=_MONIES_IMPLIED_KEEP_CANONICAL,
    bounds_drop_canonical=_MONIES_IMPLIED_BOUNDS_DROP_CANONICAL,
    bounds_null_canonical=_MONIES_IMPLIED_BOUNDS_NULL_CANONICAL,
)
