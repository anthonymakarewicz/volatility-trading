"""Row-level SOFT QC checks for ORATS option data."""

from __future__ import annotations

import polars as pl

# -----------------------------------------------------------------------------
# Strike/Maturity Arbitrage checks
# -----------------------------------------------------------------------------


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
        df_long.filter(pl.col(option_type_col) == option_type)
        .sort(g + [strike_col])
        .with_columns(
            forward_diff=(pl.col(price_col).shift(-1).over(g) - pl.col(price_col))
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
    """Flag maturity monotonicity violations (calendar arbitrage)
    for one option type.

    For fixed (trade_date, strike), option value should be non-decreasing
    with expiry (T increases).

    Violation if: price(T_next) - price(T) < -tol
    Returns subset for option_type with boolean `maturity_monot_violation`.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    g = [trade_col, strike_col]

    df_sub = (
        df_long.filter(pl.col(option_type_col) == option_type)
        .sort(g + [expiry_col])
        .with_columns(
            forward_diff=(pl.col(price_col).shift(-1).over(g) - pl.col(price_col))
        )
        .with_columns(
            maturity_monot_violation=((pl.col("forward_diff") < -tol).fill_null(False))
        )
        .drop("forward_diff")
    )
    return df_sub
