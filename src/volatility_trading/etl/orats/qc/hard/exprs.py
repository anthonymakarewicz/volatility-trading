from __future__ import annotations

import polars as pl


def expr_bad_null_keys(*keys: str) -> pl.Expr:
    """Return expression that flags rows where any key is null."""
    return pl.any_horizontal([pl.col(k).is_null() for k in keys])


def expr_bad_negative(col: str, *, eps: float = 0.0) -> pl.Expr:
    """Flag rows where col < -eps (null-safe)."""
    return pl.col(col).is_not_null() & (pl.col(col) < (0.0 - eps))


def expr_bad_trade_after_expiry(
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
) -> pl.Expr:
    """Flag rows where trade_date > expiry_date."""
    return pl.col(trade_col) > pl.col(expiry_col)


# -----------------------------------------------------------------------------
# Quote checks
# -----------------------------------------------------------------------------


def expr_bad_bid_ask(bid_col: str, ask_col: str) -> pl.Expr:
    """Flag rows where bid > ask."""
    return pl.col(bid_col) > pl.col(ask_col)


def expr_bad_negative_quotes(
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
) -> pl.Expr:
    """Rows with negative bid or ask quotes."""
    return (pl.col(bid_col) < 0) | (pl.col(ask_col) < 0)


def expr_bad_crossed_market(
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    *,
    tol: float = 0.0,
) -> pl.Expr:
    """Crossed market: bid > ask (+ optional tolerance)."""
    return pl.col(bid_col) > (pl.col(ask_col) + tol)


# -----------------------------------------------------------------------------
# Volume / Open Interest checks
# -----------------------------------------------------------------------------


def expr_bad_negative_vol_oi(
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> pl.Expr:
    """Rows where volume or open interest are negative (structurally invalid)."""
    return (pl.col(volume_col) < 0) | (pl.col(oi_col) < 0)


# -----------------------------------------------------------------------------
# Greeks checks (kept as-is; you may or may not use them yet)
# -----------------------------------------------------------------------------


def expr_bad_gamma_sign(
    gamma_col: str = "gamma",
    opt_col: str = "option_type",
    eps: float = 1e-12,
) -> pl.Expr:
    """Return expression flagging rows where gamma is invalid."""
    g = pl.col(gamma_col)
    opt = pl.col(opt_col)
    bad = g < (0.0 - eps)
    return pl.when(opt.is_in(["C", "P"])).then(bad).otherwise(True)


def expr_bad_vega_sign(
    vega_col: str = "vega",
    opt_col: str = "option_type",
    eps: float = 1e-12,
) -> pl.Expr:
    """Return expression flagging rows where vega is invalid."""
    v = pl.col(vega_col)
    opt = pl.col(opt_col)
    bad = v < (0.0 - eps)
    return pl.when(opt.is_in(["C", "P"])).then(bad).otherwise(True)
