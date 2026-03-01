"""Provider-specific options-chain mappings to canonical contract fields.

This module contains source/vendor column aliases and ETL rename maps.
Canonical field names and required/optional schema definitions remain in
``volatility_trading.contracts.options_chain``.
"""

from __future__ import annotations

from volatility_trading.contracts.options_chain import (
    ASK_PRICE,
    BID_PRICE,
    EXPIRY_DATE,
    MARKET_IV,
    MODEL_IV,
    OPEN_INTEREST,
    OPTION_TYPE,
    SPOT_PRICE,
    TRADE_DATE,
)

ORATS_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("date", "quote_date"),
    EXPIRY_DATE: ("expiry", "expire_date"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_last", "underlying_price"),
    MARKET_IV: ("market_iv", "mid_iv", "iv"),
    MODEL_IV: ("model_iv",),
    OPEN_INTEREST: ("oi",),
}

YFINANCE_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("quote_date", "date", "last_trade_date"),
    EXPIRY_DATE: ("expiration", "expiry", "expiration_date"),
    OPTION_TYPE: ("type", "option_type_label"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_price", "underlying_last"),
    MARKET_IV: ("implied_volatility", "impliedVolatility", "iv"),
    OPEN_INTEREST: ("openInterest",),
}

OPTIONSDX_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    TRADE_DATE: ("date", "quote_date", "quote_readtime"),
    EXPIRY_DATE: ("expiry", "expire_date"),
    BID_PRICE: ("bid",),
    ASK_PRICE: ("ask",),
    SPOT_PRICE: ("underlying_last", "underlying_price"),
    MARKET_IV: ("mid_iv", "iv"),
    OPEN_INTEREST: ("oi",),
}

OPTIONSDX_LONG_RENAMES_VENDOR_TO_CANONICAL: dict[str, str] = {
    "date": TRADE_DATE,
    "expiry": EXPIRY_DATE,
    "bid": BID_PRICE,
    "ask": ASK_PRICE,
    "iv": MARKET_IV,
    "underlying_last": SPOT_PRICE,
}
