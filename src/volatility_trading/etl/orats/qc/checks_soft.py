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


def summarize_by_bucket(
    df: pl.DataFrame,
    *,
    violation_col: str,
    dte_col: str = "dte",
    delta_col: str = "delta",
    dte_bins: Sequence[int | float] = (0, 10, 30, 60),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> pl.DataFrame:
    """Summarise violations by (DTE bucket, |delta| bucket)."""
    totals = df.select(
        pl.col(violation_col).sum().alias("total_viol"),
        pl.len().alias("total_rows"),
    ).row(0)
    total_viol, total_rows = totals

    if total_rows == 0:
        return pl.DataFrame()

    summary = (
        df
        .group_by(
            pl.col(dte_col).cut(dte_bins).alias("dte_bucket"),
            pl.col(delta_col).abs().cut(delta_bins).alias("delta_bucket"),
        )
        .agg(
            pl.col(violation_col).sum().alias("n_viol"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("n_viol") / pl.col("n_rows")).alias("viol_rate_bucket"),
            (pl.col("n_viol") / total_viol).alias("viol_share"),
            (pl.col("n_rows") / total_rows).alias("row_share"),
        )
        .sort("viol_rate_bucket", descending=True)
    )
    return summary