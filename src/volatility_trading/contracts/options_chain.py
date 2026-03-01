"""Canonical options-chain schema contract shared by ETL and backtesting.

This module centralizes the long-format options-chain field contract used by:
- ETL normalization/rename steps, and
- backtesting adapter normalization/validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical long-format options-chain fields
# ---------------------------------------------------------------------------

TRADE_DATE = "trade_date"
EXPIRY_DATE = "expiry_date"
DTE = "dte"
OPTION_TYPE = "option_type"
STRIKE = "strike"
DELTA = "delta"
BID_PRICE = "bid_price"
ASK_PRICE = "ask_price"
GAMMA = "gamma"
VEGA = "vega"
THETA = "theta"
SPOT_PRICE = "spot_price"
MARKET_IV = "market_iv"
MODEL_IV = "model_iv"
YTE = "yte"
OPEN_INTEREST = "open_interest"
VOLUME = "volume"

CANONICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    EXPIRY_DATE,
    DTE,
    OPTION_TYPE,
    STRIKE,
    DELTA,
    BID_PRICE,
    ASK_PRICE,
)

CANONICAL_OPTIONAL_COLUMNS: tuple[str, ...] = (
    GAMMA,
    VEGA,
    THETA,
    SPOT_PRICE,
    MARKET_IV,
    MODEL_IV,
    YTE,
    OPEN_INTEREST,
    VOLUME,
)

CANONICAL_COLUMN_SET: frozenset[str] = frozenset(
    CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS
)

CANONICAL_ALIAS_FIELDS: tuple[str, ...] = (
    TRADE_DATE,
    *CANONICAL_REQUIRED_COLUMNS,
    *CANONICAL_OPTIONAL_COLUMNS,
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    DTE,
    STRIKE,
    DELTA,
    GAMMA,
    VEGA,
    THETA,
    BID_PRICE,
    ASK_PRICE,
    SPOT_PRICE,
    MARKET_IV,
    MODEL_IV,
    YTE,
    OPEN_INTEREST,
    VOLUME,
)


# ---------------------------------------------------------------------------
# Provider alias overrides for adapter boundary
# ---------------------------------------------------------------------------

ORATS_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("date", "quote_date"),
    EXPIRY_DATE: ("expiry", "expire_date"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_last", "underlying_price"),
    MARKET_IV: ("market_iv", "mid_iv", "iv", "smoothed_iv"),
    MODEL_IV: ("model_iv", "smoothed_iv"),
    OPEN_INTEREST: ("oi",),
}

YFINANCE_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("quote_date", "date", "last_trade_date"),
    EXPIRY_DATE: ("expiration", "expiry", "expiration_date"),
    OPTION_TYPE: ("type", "option_type_label"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_price", "underlying_last"),
    MARKET_IV: ("implied_volatility", "impliedVolatility", "iv", "smoothed_iv"),
    OPEN_INTEREST: ("openInterest",),
}

OPTIONSDX_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("date", "quote_date", "quote_readtime"),
    EXPIRY_DATE: ("expiry", "expire_date"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_last", "underlying_price"),
    MARKET_IV: ("mid_iv", "iv", "smoothed_iv"),
    OPEN_INTEREST: ("oi",),
}


# ---------------------------------------------------------------------------
# Provider ETL rename maps to canonical long-format fields
# ---------------------------------------------------------------------------

OPTIONSDX_LONG_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    "date": TRADE_DATE,
    "expiry": EXPIRY_DATE,
    "bid": BID_PRICE,
    "ask": ASK_PRICE,
    "iv": MARKET_IV,
    "underlying_last": SPOT_PRICE,
}
