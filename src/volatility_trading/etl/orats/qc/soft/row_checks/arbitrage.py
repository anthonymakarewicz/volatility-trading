from __future__ import annotations

import polars as pl


# =============================================================================
# Small internal helpers
# =============================================================================

def _all_not_null(cols: list[str]) -> pl.Expr:
    """True iff all given columns are not null."""
    return pl.all_horizontal([pl.col(c).is_not_null() for c in cols])


def _all_positive(cols: list[str]) -> pl.Expr:
    """True iff all given columns are strictly > 0 (null-safe if used with _all_not_null first)."""
    return pl.all_horizontal([pl.col(c) > 0 for c in cols])


def _apply_required_violation_mask(
    *,
    required: pl.Expr,
    violation: pl.Expr,
    out_col: str,
) -> pl.Expr:
    """
    Standard pattern:
        if required then violation else False
    """
    return (
        pl.when(required)
        .then(violation)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )


def _forward_diff_over_groups(
    *,
    value_col: str,
    group_cols: list[str],
) -> pl.Expr:
    """
    Compute forward difference within groups:
        value.shift(-1).over(group) - value
    """
    return pl.col(value_col).shift(-1).over(group_cols) - pl.col(value_col)


# =============================================================================
# Put-Call Parity helpers
# =============================================================================

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


# =============================================================================
# European parity forward (equality)
# =============================================================================

def flag_put_call_parity_mid_eu_forward(
    df: pl.DataFrame,
    *,
    call_mid_col: str = "call_mid_price",
    put_mid_col: str = "put_mid_price",
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    fwd_col: str = "underlying_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    yte_col: str = "yte",
    out_col: str = "pcp_mid_eu_violation",
    multiplier: float = 1.0,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
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
        _apply_required_violation_mask(
            required=required,
            violation=violation,
            out_col=out_col,
        )
    )


# =============================================================================
# American parity (bounds)
# =============================================================================

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

    required_not_null = _all_not_null([
        spot_col,
        strike_col,
        yte_col,
        r_col,
        q_col,
        call_bid_col,
        call_mid_col,
        call_ask_col,
        put_bid_col,
        put_mid_col,
        put_ask_col,
    ])
    required_positive = _all_positive([
        call_bid_col,
        call_ask_col,
        call_mid_col,
        put_bid_col,
        put_mid_col,
        put_ask_col,
    ])
    required = required_not_null & required_positive

    return df.with_columns(
        _apply_required_violation_mask(
            required=required,
            violation=violation,
            out_col=out_col,
        )
    )


# =============================================================================
# Strike/Maturity Arbitrage checks
# =============================================================================

def flag_strike_monotonicity(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    price_col: str = "mid_price",
    tol: float = 1e-6,
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
    strike_col: str = "strike",
    option_type_col: str = "option_type",
) -> pl.DataFrame:
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    g = [trade_col, expiry_col]

    df_sub = (
        df_long
        .filter(pl.col(option_type_col) == option_type)
        .sort(g + [strike_col])
        .with_columns(
            forward_diff=_forward_diff_over_groups(value_col=price_col, group_cols=g)
        )
        .with_columns(
            strike_monot_violation=(
                pl.when(pl.lit(option_type) == "C")
                .then(pl.col("forward_diff") > tol)
                .otherwise(pl.col("forward_diff") < -tol)
                .fill_null(False)
            )
        )
        .drop("forward_diff")
    )
    return df_sub


def flag_maturity_monotonicity(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    price_col: str = "mid_price",
    tol: float = 1e-6,
    trade_col: str = "trade_date",
    strike_col: str = "strike",
    expiry_col: str = "expiry_date",
    option_type_col: str = "option_type",
) -> pl.DataFrame:
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    g = [trade_col, strike_col]

    df_sub = (
        df_long
        .filter(pl.col(option_type_col) == option_type)
        .sort(g + [expiry_col])
        .with_columns(
            forward_diff=_forward_diff_over_groups(value_col=price_col, group_cols=g)
        )
        .with_columns(
            maturity_monot_violation=(pl.col("forward_diff") < -tol).fill_null(False)
        )
        .drop("forward_diff")
    )
    return df_sub


# =============================================================================
# Theoretical Upper/Lower bounds checks
# =============================================================================

def _bounds_dyn_abs_tol(
    *,
    bid_col: str,
    ask_col: str,
    multiplier: float = 0.5,
    floor: float = 0.01,
) -> pl.Expr:
    """
    Dynamic absolute tolerance based on bid/ask spread.

    tol = max(floor, multiplier * (ask - bid))
    """
    spread = (pl.col(ask_col) - pl.col(bid_col)).abs()
    return pl.max_horizontal([pl.lit(floor), pl.lit(multiplier) * spread])


def flag_option_bounds_mid_eu_forward(
    df: pl.DataFrame,
    *,
    option_type: str,
    mid_col: str = "mid_price",
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    forward_col: str = "underlying_price",  # F0
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    yte_col: str = "yte",
    out_col: str = "price_bounds_mid_eu_violation",
    multiplier: float = 0.5,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    disc_r = (-pl.col(r_col) * pl.col(yte_col)).exp()
    is_call = pl.lit(str(option_type).upper() == "C")

    lower = pl.when(is_call).then(
        (disc_r * (pl.col(forward_col) - pl.col(strike_col))).clip(lower_bound=0.0)
    ).otherwise(
        (disc_r * (pl.col(strike_col) - pl.col(forward_col))).clip(lower_bound=0.0)
    )

    upper = pl.when(is_call).then(
        disc_r * pl.col(forward_col)
    ).otherwise(
        disc_r * pl.col(strike_col)
    )

    tol = _bounds_dyn_abs_tol(
        bid_col=bid_col,
        ask_col=ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    mid = pl.col(mid_col)
    violation = (mid < (lower - tol)) | (mid > (upper + tol))

    required = _all_not_null(
        [mid_col, bid_col, ask_col, forward_col, strike_col, r_col, yte_col]
    )

    return df.with_columns(
        _apply_required_violation_mask(
            required=required,
            violation=violation,
            out_col=out_col,
        )
    )


def flag_option_bounds_mid_am_spot(
    df: pl.DataFrame,
    *,
    option_type: str,
    mid_col: str = "mid_price",
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    spot_col: str = "spot_price",  # S0
    strike_col: str = "strike",
    out_col: str = "price_bounds_mid_am_spot_violation",
    multiplier: float = 0.5,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    is_call = pl.lit(str(option_type).upper() == "C")

    call_intrinsic = (pl.col(spot_col) - pl.col(strike_col)).clip(lower_bound=0.0)
    put_intrinsic = (pl.col(strike_col) - pl.col(spot_col)).clip(lower_bound=0.0)

    lower = pl.when(is_call).then(call_intrinsic).otherwise(put_intrinsic)
    upper = pl.when(is_call).then(pl.col(spot_col)).otherwise(pl.col(strike_col))

    tol = _bounds_dyn_abs_tol(
        bid_col=bid_col,
        ask_col=ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    mid = pl.col(mid_col)
    violation = (mid < (lower - tol)) | (mid > (upper + tol))

    required = _all_not_null([mid_col, bid_col, ask_col, spot_col, strike_col])

    return df.with_columns(
        _apply_required_violation_mask(
            required=required,
            violation=violation,
            out_col=out_col,
        )
    )