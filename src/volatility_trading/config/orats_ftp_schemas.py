"""volatility_trading.config.orats_ftp_schemas

ORATS FTP (SMV Strikes) schema definitions.

This module is intentionally **mechanical**: it defines the column contracts
needed by the ETL pipeline to ingest the ORATS-hosted FTP "smvstrikes" CSVs
and normalize them into our canonical snake_case schema.

Scope
-----
These constants are used by:
- **raw -> intermediate** extraction (vendor dtypes + vendor date parsing)
- **intermediate -> processed** builds (rename to canonical, column selection)
- optional **bounds** handling in processed (drop vs null tiers)

Naming convention
-----------------
- `STRIKES_VENDOR_*`    : raw ORATS/FTP column names (as seen in the CSVs)
- `STRIKES_CANONICAL_*` : canonical (our snake_case) column names

Notes
-----
- Vendor date columns are ingested as strings and parsed to `pl.Date` during
  normalization (FTP uses month/day/year formatting).
- Bounds are expressed in **canonical** names and are meant to prevent
  pathological values from contaminating downstream features.
"""
from __future__ import annotations

import polars as pl


# -------------------------------------------------------------------------
# Dtypes (vendor)
# -------------------------------------------------------------------------
# Dtypes are chosen to be robust for ingestion:
# - Identifiers                    -> Utf8
# - Prices / IVs / Greeks / Rates  -> Float64
# - Volumes / Open interest        -> Int64
# - Dates                          -> Utf8 (parsed later; see *_VENDOR_DATE_COLS)
#
# Rationale: the FTP CSVs can contain missing values or occasional type noise.
# Using explicit dtypes avoids costly inference and keeps the raw->intermediate
# step stable across years.

STRIKES_VENDOR_DTYPES = {
    # identifiers
    "ticker": pl.Utf8,
    "cOpra": pl.Utf8,   # OCC/OPRA call symbol
    "pOpra": pl.Utf8,   # OCC/OPRA put symbol

    # underlying / dates
    "stkPx": pl.Float64,
    "expirDate": pl.Utf8,
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

    # spot & trade date
    "spot_px": pl.Float64,
    "trade_date": pl.Utf8,
}


# -------------------------------------------------------------------------
# Date/datetime columns (vendor)
# -------------------------------------------------------------------------
# These columns are present as strings in the FTP extracts and should be
# parsed during normalization.
#
# - `trade_date` and `expirDate` are typically formatted as MM/DD/YYYY.
# - Datetime columns are not currently provided by this FTP dataset.

STRIKES_VENDOR_DATE_COLS: tuple[str, ...] = ("trade_date", "expirDate")
STRIKES_VENDOR_DATETIME_COLS: tuple[str, ...] = ()


# -------------------------------------------------------------------------
# Renames (vendor -> canonical)
# -------------------------------------------------------------------------
# Maps raw ORATS/FTP field names into our canonical snake_case schema.
#
# This is used after vendor parsing/casting. Downstream code should reference
# **canonical** names only.

STRIKES_RENAMES_VENDOR_TO_CANONICAL = {
    # identifiers
    "ticker": "ticker",
    "cOpra": "call_opra",
    "pOpra": "put_opra",

    # underlying / dates
    "stkPx": "underlying_price",  # stock/ETF: spot; index: parity-implied forward per expiry
    "spot_px": "spot_price",      # cash spot for stock/ETF and index (same across expiries)
    "expirDate": "expiry_date",
    "trade_date": "trade_date",
    "yte": "yte",
    "strike": "strike",

    # volume / open interest
    "cVolu": "call_volume",
    "cOi": "call_open_interest",
    "pVolu": "put_volume",
    "pOi": "put_open_interest",

    # quotes (prices)
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

    # greeks (raw from ORATS)
    "delta": "call_delta",
    "gamma": "call_gamma",
    "theta": "call_theta",
    "vega": "call_vega",
    "rho": "call_rho",
    "phi": "phi",
    "driftlessTheta": "driftless_theta",

    # external iv / theo values
    "extVol": "ext_iv",
    "extCTheo": "ext_call_theo",
    "extPTheo": "ext_put_theo",
}


# -------------------------------------------------------------------------
# Bounds (canonical)
# -------------------------------------------------------------------------
# Bounds are expressed in canonical column names.
#
# Purpose
# -------
# Prevent pathological values (e.g. 1e147, negative prices, etc.) from
# contaminating downstream features and joins.
#
# Two tiers
# ---------
# - *_DROP_* : out-of-bounds => drop the whole row (structural invalidity)
# - *_NULL_* : out-of-bounds => set the value to null (row can survive)
#
# Guidance
# --------
# Keep these bounds *wide* and mechanical. Strategy-specific filters belong
# in processed builders / research code.

STRIKES_BOUNDS_DROP_CANONICAL: dict[str, tuple[float, float]] = {
    # Contract / underlying validity
    "underlying_price": (0.0, 1e7),
    "spot_price": (0.0, 1e7),
    "strike": (0.0, 1e9),
    "yte": (0.0, 10.0),
    "dte": (0.0, 3650.0),

    # Quote integrity (negative prices are invalid; absurdly large prices too)
    "call_bid_price": (0.0, 1e7),
    "call_ask_price": (0.0, 1e7),
    "put_bid_price": (0.0, 1e7),
    "put_ask_price": (0.0, 1e7),
}


STRIKES_BOUNDS_NULL_CANONICAL: dict[str, tuple[float, float]] = {
    # Volumes / open interest (null if absurd)
    "call_volume": (0.0, 1e9),
    "put_volume": (0.0, 1e9),
    "call_open_interest": (0.0, 1e9),
    "put_open_interest": (0.0, 1e9),

    # Prices / derived price features (null if absurd)
    "call_model_price": (0.0, 1e7),
    "put_model_price": (0.0, 1e7),
    "call_mid_price": (0.0, 1e7),
    "put_mid_price": (0.0, 1e7),
    "call_spread": (0.0, 1e7),
    "put_spread": (0.0, 1e7),
    "call_rel_spread": (0.0, 10.0),
    "put_rel_spread": (0.0, 10.0),

    # Moneyness (dimensionless)
    "moneyness_ks": (0.0, 1000.0),

    # Implied vols
    "smoothed_iv": (0.0, 10.0),
    "call_bid_iv": (0.0, 10.0),
    "call_mid_iv": (0.0, 10.0),
    "call_ask_iv": (0.0, 10.0),
    "put_bid_iv": (0.0, 10.0),
    "put_mid_iv": (0.0, 10.0),
    "put_ask_iv": (0.0, 10.0),

    # Rates / yields
    "risk_free_rate": (-1.0, 1.0),
    "dividend_yield": (-5.0, 5.0),
    "residual_rate_data": (-5.0, 5.0),

    # Greeks (wide bounds; aim is to kill absurd values, not do strategy QC)
    "call_delta": (-5.0, 5.0),
    "put_delta": (-5.0, 5.0),
    "call_gamma": (-1000.0, 1000.0),
    "put_gamma": (-1000.0, 1000.0),
    "call_theta": (-1e6, 1e6),
    "put_theta": (-1e6, 1e6),
    "call_vega": (-1e6, 1e6),
    "put_vega": (-1e6, 1e6),
    "call_rho": (-1e6, 1e6),
    "put_rho": (-1e6, 1e6),
    "phi": (-1e6, 1e6),
    "driftless_theta": (-1e6, 1e6),

    # External vol / theo
    "ext_iv": (0.0, 10.0),
    "ext_call_theo": (0.0, 1e7),
    "ext_put_theo": (0.0, 1e7),
}


# -------------------------------------------------------------------------
# Keep (canonical)
# -------------------------------------------------------------------------
# Canonical columns retained in the processed options-chain output.
#
# Notes:
# - This list is intentionally curated (not "keep everything").
# - Additive growth is fine: when you add new canonical columns, you can
#   update this list and (optionally) extend bounds to cover them.

STRIKES_KEEP_CANONICAL = [
    # identifiers / dates
    "ticker",
    "trade_date",
    "expiry_date",
    "dte",
    "yte",

    # underlying & strike
    "underlying_price",
    "spot_price",
    "strike",

    # volume & open interest
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",

    # prices
    "call_bid_price",
    "call_mid_price",
    "call_model_price",
    "call_ask_price",
    "call_rel_spread",
    "put_bid_price",
    "put_mid_price",
    "put_model_price",
    "put_ask_price",
    "put_rel_spread",

    # main vols
    "smoothed_iv",
    "call_mid_iv",
    "put_mid_iv",

    # Greeks (already split C/P)
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_vega",
    "call_rho",
    "put_delta",
    "put_gamma",
    "put_theta",
    "put_vega",
    "put_rho",

    # curves
    "risk_free_rate",
    "dividend_yield",
]