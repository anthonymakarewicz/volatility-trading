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
