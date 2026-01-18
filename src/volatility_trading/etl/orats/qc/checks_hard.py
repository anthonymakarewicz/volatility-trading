from __future__ import annotations

import polars as pl


def expr_bad_null_keys(*keys: str) -> pl.Expr:
    """Return expression that flags rows where any key is null."""
    return pl.any_horizontal([pl.col(k).is_null() for k in keys])


def expr_bad_bid_ask(bid_col: str, ask_col: str) -> pl.Expr:
    """Flag rows where bid > ask."""
    return pl.col(bid_col) > pl.col(ask_col)


def expr_bad_trade_after_expiry(
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
) -> pl.Expr:
    """Flag rows where trade_date > expiry_date."""
    return pl.col(trade_col) > pl.col(expiry_col)


def expr_bad_negative(col: str) -> pl.Expr:
    """Flag rows where col < 0."""
    return pl.col(col) < 0