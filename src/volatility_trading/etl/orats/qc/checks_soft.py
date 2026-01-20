from __future__ import annotations

from collections.abc import Sequence

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


def flag_rho_wrong_sign(
    df: pl.DataFrame,
    option_type: str,
    *,
    rho_col: str = "rho",
    eps: float = 1e-8,
    out_col: str = "rho_wrong_sign_violation",
) -> pl.DataFrame:
    """Flag rho with an unexpected sign (diagnostic).

    Typical expectation (European intuition):
    - call_rho >= 0
    - put_rho  <= 0

    Treat as SOFT since vendors/models can differ slightly.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")
    
    rho = pl.col(rho_col)
    bad = rho < -eps if option_type == "C" else rho > eps

    return df.filter(pl.col("option_type") == option_type).with_columns(
        bad.fill_null(False).alias(out_col)
    )