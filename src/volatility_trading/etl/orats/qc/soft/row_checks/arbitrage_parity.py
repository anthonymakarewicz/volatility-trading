"""Row-level SOFT QC checks for ORATS option data."""

from __future__ import annotations

import polars as pl

from .expr_helpers import _all_not_null, _apply_required_mask


def _pcp_dyn_abs_tol(
    *,
    call_bid_col: str,
    call_ask_col: str,
    put_bid_col: str,
    put_ask_col: str,
    multiplier: float = 0.5,
    floor: float = 0.0,
) -> pl.Expr:
    """Dynamic absolute tolerance for MID-price parity checks derived from spreads."""
    call_spread = pl.col(call_ask_col) - pl.col(call_bid_col)
    put_spread = pl.col(put_ask_col) - pl.col(put_bid_col)

    return multiplier * (call_spread + put_spread) + floor


def _pcp_discounts(
    *,
    yte_col: str,
    r_col: str,
    q_col: str,
) -> tuple[pl.Expr, pl.Expr]:
    """Return (disc_q, disc_r) expressions."""
    disc_q = (-pl.col(q_col) * pl.col(yte_col)).exp()
    disc_r = (-pl.col(r_col) * pl.col(yte_col)).exp()
    return disc_q, disc_r


# ---- European parity forward (equality) -----


def flag_put_call_parity_mid_eu_forward(
    df: pl.DataFrame,
    *,
    call_mid_col: str = "call_mid_price",
    put_mid_col: str = "put_mid_price",
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    fwd_col: str = "underlying_price",  # ORATS implied forward
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    yte_col: str = "yte",
    out_col: str = "pcp_mid_eu_violation",
    multiplier: float = 1.0,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    """
    European put-call parity using MID prices (forward form):

        C_mid - P_mid == exp(-rT) * (F - K)

    where F is the implied forward price (here: `fwd_col`).

    Violation rule:
        |lhs - rhs| > dyn_tol

    dyn_tol is derived from bid/ask spreads with an optional floor.
    """
    disc_r = (-pl.col(r_col) * pl.col(yte_col)).exp()

    rhs = disc_r * (pl.col(fwd_col) - pl.col(strike_col))
    lhs = pl.col(call_mid_col) - pl.col(put_mid_col)

    dyn_tol = _pcp_dyn_abs_tol(
        call_bid_col=call_bid_col,
        call_ask_col=call_ask_col,
        put_bid_col=put_bid_col,
        put_ask_col=put_ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    violation = (lhs - rhs).abs() > dyn_tol

    required = _all_not_null(
        [
            call_mid_col,
            put_mid_col,
            fwd_col,
            strike_col,
            yte_col,
            r_col,
            call_bid_col,
            call_ask_col,
            put_bid_col,
            put_ask_col,
        ]
    )

    return df.with_columns(
        _apply_required_mask(
            violation_expr=violation,
            required_expr=required,
            out_col=out_col,
        )
    )


# ---- American parity (bounds) -----


def flag_put_call_parity_bounds_mid_am(
    df: pl.DataFrame,
    *,
    call_mid_col: str = "call_mid_price",
    put_mid_col: str = "put_mid_price",
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    spot_col: str = "spot_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    yte_col: str = "yte",
    out_col: str = "pcp_bounds_mid_am_violation",
    multiplier: float = 0.5,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    """
    American put-call parity bounds using MID prices (Hull, with dividend yield q):

        S0*e^{-qT} - K  <=  C_A - P_A  <=  S0 - K*e^{-rT}

    We flag a violation when the MID difference is outside the bounds by more
    than a dynamic absolute tolerance derived from bid/ask spreads:

        lhs < lower - dyn_tol  OR  lhs > upper + dyn_tol
    """
    disc_q, disc_r = _pcp_discounts(yte_col=yte_col, r_col=r_col, q_col=q_col)

    lhs = pl.col(call_mid_col) - pl.col(put_mid_col)

    lower = pl.col(spot_col) * disc_q - pl.col(strike_col)
    upper = pl.col(spot_col) - pl.col(strike_col) * disc_r

    dyn_tol = _pcp_dyn_abs_tol(
        call_bid_col=call_bid_col,
        call_ask_col=call_ask_col,
        put_bid_col=put_bid_col,
        put_ask_col=put_ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    violation = (lhs < (lower - dyn_tol)) | (lhs > (upper + dyn_tol))

    required = pl.all_horizontal(
        [
            pl.col(spot_col).is_not_null(),
            pl.col(strike_col).is_not_null(),
            pl.col(yte_col).is_not_null(),
            pl.col(r_col).is_not_null(),
            pl.col(q_col).is_not_null(),
            pl.col(call_bid_col).is_not_null(),
            pl.col(call_mid_col).is_not_null(),
            pl.col(call_ask_col).is_not_null(),
            pl.col(put_bid_col).is_not_null(),
            pl.col(put_mid_col).is_not_null(),
            pl.col(put_ask_col).is_not_null(),
            pl.col(call_bid_col) > 0,
            pl.col(call_ask_col) > 0,
            pl.col(call_mid_col) > 0,
            pl.col(put_bid_col) > 0,
            pl.col(put_mid_col) > 0,
            pl.col(put_ask_col) > 0,
        ]
    )

    return df.with_columns(
        _apply_required_mask(
            violation_expr=violation,
            required_expr=required,
            out_col=out_col,
        )
    )
