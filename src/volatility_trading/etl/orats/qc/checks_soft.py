from __future__ import annotations

import polars as pl


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
    """Flag strike monotonicity violations for one option type.

    Within each (trade_date, expiry_date), sorted by strike:
      - Calls: price must be non-increasing in strike.
      - Puts : price must be non-decreasing in strike.

    We flag a violation when the forward difference breaks monotonicity:
      - Calls: price(K_next) - price(K) >  tol
      - Puts : price(K_next) - price(K) < -tol

    Returns a subset for the requested option_type with boolean column
    `strike_monot_violation`.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    g = [trade_col, expiry_col]

    df_sub = (
        df_long
        .filter(pl.col(option_type_col) == option_type)
        .sort(g + [strike_col])
        .with_columns(
            forward_diff=(
                pl.col(price_col).shift(-1).over(g) - pl.col(price_col)
            )
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
    """Flag maturity monotonicity violations (calendar arbitrage) for one option type.

    For fixed (trade_date, strike), option value should be non-decreasing
    with expiry (T increases).

    Violation if: price(T_next) - price(T) < -tol
    Returns subset for option_type with boolean `maturity_monot_violation`.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    g = [trade_col, strike_col]

    df_sub = (
        df_long
        .filter(pl.col(option_type_col) == option_type)
        .sort(g + [expiry_col])
        .with_columns(
            forward_diff=(
                pl.col(price_col).shift(-1).over(g) - pl.col(price_col)
            )
        )
        .with_columns(
            maturity_monot_violation=(
                (pl.col("forward_diff") < -tol).fill_null(False)
            )
        )
        .drop("forward_diff")
    )
    return df_sub


def flag_locked_market(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    out_col: str = "locked_market_violation",
) -> pl.DataFrame:
    """Locked market: bid == ask > 0 (liquidity diagnostic)."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return (
        df_long
        .filter(pl.col("option_type") == option_type)
        .with_columns(
            ((pl.col(bid_col) == pl.col(ask_col)) & (pl.col(bid_col) > 0))
            .fill_null(False)
            .alias(out_col)
        )
    )


def flag_one_sided_quotes(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    out_col: str = "one_sided_quote_violation",
) -> pl.DataFrame:
    """One-sided quote: bid == 0 and ask > 0."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return (
        df_long
        .filter(pl.col("option_type") == option_type)
        .with_columns(
            ((pl.col(bid_col) == 0) & (pl.col(ask_col) > 0))
            .fill_null(False)
            .alias(out_col)
        )
    )


def flag_wide_spread(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    rel_spread_col: str = "rel_spread",
    mid_col: str = "mid_price",
    min_mid: float = 0.01,
    threshold: float = 1.0,
    out_col: str = "wide_spread_violation",
) -> pl.DataFrame:
    """Wide spread flag: rel_spread > threshold, only when mid >= min_mid."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    df_sub = df_long.filter(pl.col("option_type") == option_type)

    return df_sub.with_columns(
        pl.when(pl.col(mid_col) >= min_mid)
        .then(pl.col(rel_spread_col) > threshold)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )


# ----- Volume / Open Interest checks -----

def flag_zero_volume(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Flag rows where volume == 0."""
    df_sub = (
        df.filter(pl.col("option_type") == option_type)
        .with_columns(
            zero_volume_violation=(pl.col(volume_col) == 0).fill_null(False)
        )
    )
    return df_sub


def flag_zero_open_interest(
    df: pl.DataFrame,
    option_type: str,
    *,
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where open interest == 0."""
    df_sub = (
        df.filter(pl.col("option_type") == option_type)
        .with_columns(
            zero_open_interest_violation=(pl.col(oi_col) == 0).fill_null(False)
        )
    )
    return df_sub


def flag_zero_vol_pos_oi(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where volume == 0 but OI > 0."""
    df_sub = (
        df.filter(pl.col("option_type") == option_type)
        .with_columns(
            zero_vol_pos_oi_violation=(
                (pl.col(volume_col) == 0) & (pl.col(oi_col) > 0)
            ).fill_null(False)
        )
    )
    return df_sub


def flag_pos_vol_zero_oi(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where volume > 0 but OI == 0."""
    df_sub = (
        df.filter(pl.col("option_type") == option_type)
        .with_columns(
            pos_vol_zero_oi_violation=(
                (pl.col(volume_col) > 0) & (pl.col(oi_col) == 0)
            ).fill_null(False)
        )
    )
    return df_sub


# ----- Greeks -----

def flag_theta_positive(
    df: pl.DataFrame,
    option_type: str,
    *,
    theta_col: str = "theta",
    eps: float = 1e-8,
    out_col: str = "theta_positive_violation",
) -> pl.DataFrame:
    """Flag positive theta beyond tolerance (diagnostic).

    Positive theta can happen legitimately (dividends/carry), so this is SOFT.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return df.filter(pl.col("option_type") == option_type).with_columns(
        (pl.col(theta_col) > eps).fill_null(False).alias(out_col)
    )


def flag_iv_high(
    df: pl.DataFrame,
    *,
    iv_col: str = "smoothed_iv",
    threshold: float = 1.0,
    out_col: str = "iv_too_high_violation",
) -> pl.DataFrame:
    """Flag rows where IV is above a threshold (null-safe)."""
    return df.with_columns(
        (pl.col(iv_col).is_not_null() & (pl.col(iv_col) > threshold))
        .fill_null(False)
        .alias(out_col)
    )


import polars as pl


# -----------------------------------------------------------------------------
# Put-Call Parity helpers
# -----------------------------------------------------------------------------

def _pcp_dyn_abs_tol_from_spread(
    *,
    call_bid_col: str,
    call_ask_col: str,
    put_bid_col: str,
    put_ask_col: str,
    floor: float = 0.0,
) -> pl.Expr:
    """
    Dynamic absolute tolerance for MID-price parity checks derived from spreads.

    We use:
        tol = 0.5 * (call_spread + put_spread) + floor
    where:
        call_spread = call_ask - call_bid
        put_spread  = put_ask  - put_bid

    This is a pragmatic proxy for "mid uncertainty" due to quoting width.
    """
    call_spread = (pl.col(call_ask_col) - pl.col(call_bid_col))
    put_spread = (pl.col(put_ask_col) - pl.col(put_bid_col))

    return (1 * (call_spread + put_spread) + floor)


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


# -----------------------------------------------------------------------------
# European parity (equality)
# -----------------------------------------------------------------------------

def flag_put_call_parity_mid_eu(
    df: pl.DataFrame,
    *,
    call_mid_col: str = "call_mid_price",
    put_mid_col: str = "put_mid_price",
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    spot_col: str = "underlying_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    yte_col: str = "yte",
    out_col: str = "pcp_mid_eu_violation",
    tol_floor: float = 0.0,
) -> pl.DataFrame:
    """
    European put-call parity using MID prices:

        C_mid - P_mid == S*e^{-qT} - K*e^{-rT}

    We flag a violation when:
        |lhs - rhs| > dyn_tol
    where dyn_tol is derived from bid/ask spreads.
    """
    disc_q, disc_r = _pcp_discounts(yte_col=yte_col, r_col=r_col, q_col=q_col)

    rhs = pl.col(spot_col) * disc_q - pl.col(strike_col) * disc_r
    lhs = pl.col(call_mid_col) - pl.col(put_mid_col)

    dyn_tol = _pcp_dyn_abs_tol_from_spread(
        call_bid_col=call_bid_col,
        call_ask_col=call_ask_col,
        put_bid_col=put_bid_col,
        put_ask_col=put_ask_col,
        floor=tol_floor,
    )

    violation = (lhs - rhs).abs() > dyn_tol

    return df.with_columns(
        pl.when(
            pl.all_horizontal(
                [
                    pl.col(call_mid_col).is_not_null(),
                    pl.col(put_mid_col).is_not_null(),
                    pl.col(spot_col).is_not_null(),
                    pl.col(strike_col).is_not_null(),
                    pl.col(yte_col).is_not_null(),
                    pl.col(r_col).is_not_null(),
                    pl.col(q_col).is_not_null(),
                    pl.col(call_bid_col).is_not_null(),
                    pl.col(call_ask_col).is_not_null(),
                    pl.col(put_bid_col).is_not_null(),
                    pl.col(put_ask_col).is_not_null(),
                ]
            )
        )
        .then(violation)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )


def flag_put_call_parity_tradable_eu(
    df: pl.DataFrame,
    *,
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    spot_col: str = "underlying_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    yte_col: str = "yte",
    abs_tol: float = 0.01,
    out_col: str = "pcp_tradable_eu_violation",
) -> pl.DataFrame:
    """
    European parity tradable containment check.

    We verify that the theoretical forward PV(F-K) lies INSIDE the tradable
    interval implied by bid/ask:

        rhs = S*e^{-qT} - K*e^{-rT}

        lower = C_bid - P_ask
        upper = C_ask - P_bid

    Condition (with a tiny cushion abs_tol):
        rhs >= lower - abs_tol
        rhs <= upper + abs_tol
    """
    disc_q, disc_r = _pcp_discounts(yte_col=yte_col, r_col=r_col, q_col=q_col)

    rhs = pl.col(spot_col) * disc_q - pl.col(strike_col) * disc_r
    lower = pl.col(call_bid_col) - pl.col(put_ask_col)
    upper = pl.col(call_ask_col) - pl.col(put_bid_col)

    violation = (rhs < (lower - abs_tol)) | (rhs > (upper + abs_tol))

    return df.with_columns(
        pl.when(
            pl.all_horizontal(
                [
                    pl.col(call_bid_col).is_not_null(),
                    pl.col(call_ask_col).is_not_null(),
                    pl.col(put_bid_col).is_not_null(),
                    pl.col(put_ask_col).is_not_null(),
                    pl.col(spot_col).is_not_null(),
                    pl.col(strike_col).is_not_null(),
                    pl.col(yte_col).is_not_null(),
                    pl.col(r_col).is_not_null(),
                    pl.col(q_col).is_not_null(),
                ]
            )
        )
        .then(violation)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )


# -----------------------------------------------------------------------------
#  American parity (bounds)
# -----------------------------------------------------------------------------

def flag_put_call_parity_bounds_mid_am(
    df: pl.DataFrame,
    *,
    call_mid_col: str = "call_mid_price",
    put_mid_col: str = "put_mid_price",
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    spot_col: str = "underlying_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    yte_col: str = "yte",
    out_col: str = "pcp_bounds_mid_am_violation",
    tol_floor: float = 0.0,
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

    dyn_tol = _pcp_dyn_abs_tol_from_spread(
        call_bid_col=call_bid_col,
        call_ask_col=call_ask_col,
        put_bid_col=put_bid_col,
        put_ask_col=put_ask_col,
        floor=tol_floor,
    )

    violation = (lhs < (lower - dyn_tol)) | (lhs > (upper + dyn_tol))

    required = pl.all_horizontal(
        [
            pl.col(call_mid_col).is_not_null(),
            pl.col(put_mid_col).is_not_null(),
            pl.col(spot_col).is_not_null(),
            pl.col(strike_col).is_not_null(),
            pl.col(yte_col).is_not_null(),
            pl.col(r_col).is_not_null(),
            pl.col(q_col).is_not_null(),
            pl.col(call_bid_col).is_not_null(),
            pl.col(call_ask_col).is_not_null(),
            pl.col(put_bid_col).is_not_null(),
            pl.col(put_ask_col).is_not_null(),
        ]
    )

    return df.with_columns(
        pl.when(required)
        .then(violation)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )


def flag_put_call_parity_bounds_tradable_am(
    df: pl.DataFrame,
    *,
    call_bid_col: str = "call_bid_price",
    call_ask_col: str = "call_ask_price",
    put_bid_col: str = "put_bid_price",
    put_ask_col: str = "put_ask_price",
    spot_col: str = "underlying_price",
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    q_col: str = "dividend_yield",
    yte_col: str = "yte",
    abs_tol: float = 0.01,
    out_col: str = "pcp_bounds_tradable_am_violation",
) -> pl.DataFrame:
    """
    American put-call parity *bounds* (with dividends) using tradable bid/ask
    worst-case checks (Hull):

        S0*e^{-qT} - K <= C_A - P_A <= S0 - K*e^{-rT}

    Tradable worst-case constraints:
        lower_worst = C_bid - P_ask  >= (S0*e^{-qT} - K) - abs_tol
        upper_worst = C_ask - P_bid  <= (S0 - K*e^{-rT}) + abs_tol
    """
    disc_q, disc_r = _pcp_discounts(yte_col=yte_col, r_col=r_col, q_col=q_col)

    lower_bound = pl.col(spot_col) * disc_q - pl.col(strike_col)
    upper_bound = pl.col(spot_col) - pl.col(strike_col) * disc_r

    lower_worst = pl.col(call_bid_col) - pl.col(put_ask_col)
    upper_worst = pl.col(call_ask_col) - pl.col(put_bid_col)

    violation = (lower_worst < (lower_bound - abs_tol)) | (
        upper_worst > (upper_bound + abs_tol)
    )

    required = pl.all_horizontal(
        [
            pl.col(call_bid_col).is_not_null(),
            pl.col(call_ask_col).is_not_null(),
            pl.col(put_bid_col).is_not_null(),
            pl.col(put_ask_col).is_not_null(),
            pl.col(spot_col).is_not_null(),
            pl.col(strike_col).is_not_null(),
            pl.col(yte_col).is_not_null(),
            pl.col(r_col).is_not_null(),
            pl.col(q_col).is_not_null(),
        ]
    )

    return df.with_columns(
        pl.when(required)
        .then(violation)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )