"""
ORATS FTP (SMV Strikes) schema spec.

This spec supports mechanical normalization for the FTP strikes dataset:
- cast vendor fields to stable dtypes (pre-rename)
- parse vendor date fields (pre-rename)
- rename vendor -> canonical
- select canonical columns to keep (intermediate)
- bounds:
    * bounds_drop_canonical: out-of-bounds => drop row (structural validity)
    * bounds_null_canonical: out-of-bounds => set to null

Notes
-----
- FTP strikes dates are MM/DD/YYYY (vendor strings).
- OPRA codes appear around ~2010; earlier years may have them missing.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from .schema_spec import OratsSchemaSpec


# ----------------------------------------------------------------------------
# Dtypes (vendor)
# ----------------------------------------------------------------------------
# Dtypes are chosen to be robust for ingestion:
# - Identifiers                    -> Utf8
# - Prices / IVs / Greeks / Rates  -> Float64
# - Volumes / Open interest        -> Int64
# - Dates                          -> Utf8 (parsed later; see *_VENDOR_DATE_COLS)
#
# Rationale: the FTP CSVs can contain missing values or occasional type noise.
# Using explicit dtypes avoids costly inference and keeps the raw->intermediate
# step stable across years.

_STRIKES_VENDOR_DTYPES: dict[str, pl.DataType] = {
    # identifiers
    "ticker": pl.Utf8,
    "cOpra": pl.Utf8,
    "pOpra": pl.Utf8,

    # underlying / dates
    "stkPx": pl.Float64,
    "spot_px": pl.Float64,
    "expirDate": pl.Utf8,
    "trade_date": pl.Utf8,
    "yte": pl.Float64,
    "strike": pl.Float64,

    # volume / open interest
    "cVolu": pl.Int64,
    "cOi": pl.Int64,
    "pVolu": pl.Int64,
    "pOi": pl.Int64,

    # quotes
    "cBidPx": pl.Float64,
    "cValue": pl.Float64,
    "cAskPx": pl.Float64,
    "pBidPx": pl.Float64,
    "pValue": pl.Float64,
    "pAskPx": pl.Float64,

    # implied vols
    "cBidIv": pl.Float64,
    "cMidIv": pl.Float64,
    "cAskIv": pl.Float64,
    "smoothSmvVol": pl.Float64,
    "pBidIv": pl.Float64,
    "pMidIv": pl.Float64,
    "pAskIv": pl.Float64,

    # rates
    "iRate": pl.Float64,
    "divRate": pl.Float64,
    "residualRateData": pl.Float64,

    # greeks
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "theta": pl.Float64,
    "vega": pl.Float64,
    "rho": pl.Float64,
    "phi": pl.Float64,
    "driftlessTheta": pl.Float64,

    # external vols / theo
    "extVol": pl.Float64,
    "extCTheo": pl.Float64,
    "extPTheo": pl.Float64,
}


# -------------------------------------------------------------------------
# Date/datetime columns (vendor)
# -------------------------------------------------------------------------
# These columns are present as strings in the FTP extracts and should be
# parsed during normalization.
#
# - `trade_date` and `expirDate` are typically formatted as MM/DD/YYYY.
# - Datetime columns are not currently provided by this FTP dataset.

_STRIKES_VENDOR_DATE_COLS: tuple[str, ...] = ("trade_date", "expirDate")
_STRIKES_VENDOR_DATETIME_COLS: tuple[str, ...] = ()


# -------------------------------------------------------------------------
# Renames (vendor -> canonical)
# -------------------------------------------------------------------------
# Maps raw ORATS/FTP field names into our canonical snake_case schema.
#
# This is used after vendor parsing/casting. Downstream code should reference
# **canonical** names only.

_STRIKES_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    # identifiers
    "ticker": "ticker",
    "cOpra": "call_opra",
    "pOpra": "put_opra",

    # underlying / dates
    "stkPx": "underlying_price",
    "spot_px": "spot_price",
    "expirDate": "expiry_date",
    "trade_date": "trade_date",
    "yte": "yte",
    "strike": "strike",

    # volume / open interest
    "cVolu": "call_volume",
    "cOi": "call_open_interest",
    "pVolu": "put_volume",
    "pOi": "put_open_interest",

    # quotes
    "cBidPx": "call_bid_price",
    "cValue": "call_model_price",
    "cAskPx": "call_ask_price",
    "pBidPx": "put_bid_price",
    "pValue": "put_model_price",
    "pAskPx": "put_ask_price",

    # implied vols
    "cBidIv": "call_bid_iv",
    "cMidIv": "call_mid_iv",
    "cAskIv": "call_ask_iv",
    "smoothSmvVol": "smoothed_iv",
    "pBidIv": "put_bid_iv",
    "pMidIv": "put_mid_iv",
    "pAskIv": "put_ask_iv",

    # rates
    "iRate": "risk_free_rate",
    "divRate": "dividend_yield",
    "residualRateData": "residual_rate_data",

    # greeks (raw)
    "delta": "call_delta",
    "gamma": "call_gamma",
    "theta": "call_theta",
    "vega": "call_vega",
    "rho": "call_rho",
    "phi": "phi",
    "driftlessTheta": "driftless_theta",

    # external
    "extVol": "ext_iv",
    "extCTheo": "ext_call_theo",
    "extPTheo": "ext_put_theo",
}


# ----------------------------------------------------------------------------
# Keep (canonical) - intermediate strikes
# ----------------------------------------------------------------------------

_STRIKES_KEEP_CANONICAL: tuple[str, ...] = (
    # identifiers / dates
    "ticker",
    "trade_date",
    "expiry_date",
    "yte",
    "strike",

    # underlying
    "underlying_price",
    "spot_price",

    # identifiers (useful for de-dupe; optional but I’d keep)
    "call_opra",
    "put_opra",

    # volume / OI
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",

    # quotes
    "call_bid_price",
    "call_model_price",
    "call_ask_price",
    "put_bid_price",
    "put_model_price",
    "put_ask_price",

    # vols
    "smoothed_iv",
    "call_mid_iv",
    "put_mid_iv",

    # curves
    "risk_free_rate",
    "dividend_yield",
    "residual_rate_data",

    # call greeks (vendor only)
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_vega",
    "call_rho",

    # optional vendor-only columns if you actually use them downstream
    # "phi",
    # "driftless_theta",
    # "ext_iv",
    # "ext_call_theo",
    # "ext_put_theo",
)


# ----------------------------------------------------------------------------
# Bounds (canonical)
# ----------------------------------------------------------------------------

# Tier 1: structural validity => drop row
_STRIKES_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {
    "underlying_price": (0.0, 1e7),
    "spot_price": (0.0, 1e7),
    "strike": (0.0, 1e9),
    "yte": (0.0, 10.0),
}

# Tier 2: value plausibility => set to null
_STRIKES_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    # prices
    "call_bid_price": (0.0, 1e7),
    "call_model_price": (0.0, 1e7),
    "call_ask_price": (0.0, 1e7),
    "put_bid_price": (0.0, 1e7),
    "put_model_price": (0.0, 1e7),
    "put_ask_price": (0.0, 1e7),

    # vols
    "smoothed_iv": (0.0, 10.0),
    "call_bid_iv": (0.0, 10.0),
    "call_mid_iv": (0.0, 10.0),
    "call_ask_iv": (0.0, 10.0),
    "put_bid_iv": (0.0, 10.0),
    "put_mid_iv": (0.0, 10.0),
    "put_ask_iv": (0.0, 10.0),

    # rates / yields
    "risk_free_rate": (-1.0, 1.0),
    "dividend_yield": (-1.0, 1.0),
    "residual_rate_data": (-2.0, 2.0),

    # volumes / OI (negative makes no sense; keep row but null the field)
    "call_volume": (0.0, 1e9),
    "put_volume": (0.0, 1e9),
    "call_open_interest": (0.0, 1e9),
    "put_open_interest": (0.0, 1e9),

    # greeks (very wide; null absurds)
    "call_delta": (-2.0, 2.0),
    "call_gamma": (0.0, 1e3),
    "call_theta": (-1e5, 1e5),
    "call_vega": (0.0, 1e5),
    "call_rho": (-1e5, 1e5),

    # misc
    "phi": (-1e5, 1e5),
    "driftless_theta": (-1e5, 1e5),
    "ext_iv": (0.0, 10.0),
    "ext_call_theo": (0.0, 1e7),
    "ext_put_theo": (0.0, 1e7),
}


# ----------------------------------------------------------------------------
# Public spec object
# ----------------------------------------------------------------------------

STRIKES_SCHEMA_SPEC: Final[OratsSchemaSpec] = OratsSchemaSpec(
    vendor_dtypes=_STRIKES_VENDOR_DTYPES,
    vendor_date_cols=_STRIKES_VENDOR_DATE_COLS,
    vendor_datetime_cols=_STRIKES_VENDOR_DATETIME_COLS,
    renames_vendor_to_canonical=_STRIKES_RENAMES_VENDOR_TO_CANONICAL,
    keep_canonical=_STRIKES_KEEP_CANONICAL,
    bounds_drop_canonical=_STRIKES_BOUNDS_DROP_CANONICAL,
    bounds_null_canonical=_STRIKES_BOUNDS_NULL_CANONICAL,
)