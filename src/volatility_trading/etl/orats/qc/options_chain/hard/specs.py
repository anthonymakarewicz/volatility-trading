from __future__ import annotations

from ...hard.exprs import (
    expr_bad_bid_ask,
    expr_bad_crossed_market,
    expr_bad_negative,
    expr_bad_negative_quotes,
    expr_bad_negative_vol_oi,
    expr_bad_null_keys,
    expr_bad_trade_after_expiry,
)
from ...hard.spec_types import HardSpec


BASE_KEYS = [
    "trade_date",
    "expiry_date",
    "option_type",
    "underlying_price",
    "strike",
]


def get_hard_specs() -> list[HardSpec]:
    """Return HARD (must-pass) check specs for the options chain QC."""
    return [
        # ---- Keys / dates ----
        HardSpec(
            name="keys_not_null",
            predicate_expr=expr_bad_null_keys(
                "trade_date", "expiry_date", "strike"
            ),
            sample_cols=BASE_KEYS,
        ),
        HardSpec(
            name="trade_date_leq_expiry_date",
            predicate_expr=expr_bad_trade_after_expiry(),
            sample_cols=["trade_date", "expiry_date", "strike", "option_type"],
        ),
        # ---- Quote diagnostics ----
        HardSpec(
            name="bid_ask_sane",
            predicate_expr=expr_bad_bid_ask("bid_price", "ask_price"),
            sample_cols=BASE_KEYS + ["bid_price", "ask_price", "mid_price"],
        ),
        HardSpec(
            name="negative_quotes",
            predicate_expr=expr_bad_negative_quotes("bid_price", "ask_price"),
            sample_cols=BASE_KEYS + ["bid_price", "ask_price", "mid_price"],
        ),
        HardSpec(
            name="crossed_market",
            predicate_expr=expr_bad_crossed_market("bid_price", "ask_price"),
            sample_cols=BASE_KEYS + ["bid_price", "ask_price", "mid_price"],
        ),
        # ---- Volume / OI diagnostics ----
        HardSpec(
            name="negative_vol_oi",
            predicate_expr=expr_bad_negative_vol_oi("volume", "open_interest"),
            sample_cols=BASE_KEYS + ["volume", "open_interest"],
        ),
        # ---- Greeks sign diagnostics ----
        HardSpec(
            name="gamma_non_negative",
            predicate_expr=expr_bad_negative("gamma", eps=1e-8),
            sample_cols=BASE_KEYS + ["gamma"],
        ),
        HardSpec(
            name="vega_non_negative",
            predicate_expr=expr_bad_negative("vega", eps=1e-8),
            sample_cols=BASE_KEYS + ["vega"],
        ),
        # ---- IV sign diagnostics ----
        HardSpec(
            name="iv_non_negative",
            predicate_expr=expr_bad_negative("smoothed_iv", eps=1e-5),
            sample_cols=BASE_KEYS + ["smoothed_iv"],
        ),
    ]
