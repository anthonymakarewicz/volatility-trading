"""volatility_trading.etl.orats.processed.options_chain.config

Configuration constants for the ORATS processed options-chain builder.

Notes
-----
We keep one shared logger name across the split modules so logs still appear
under the same name as before (helps continuity in your logs).
"""

from __future__ import annotations


# Output schema (processed dataset)
OPTIONS_CHAIN_CORE_COLUMNS: tuple[str, ...] = (
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

    # vols
    "smoothed_iv",
    "call_mid_iv",
    "put_mid_iv",

    # greeks (C + parity-derived P)
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
)