from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def flag_strike_monotonicity_long(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    price_col: str = "mid_price",
    tol: float = 1e-6,
) -> pl.DataFrame:
    """
    Add a boolean 'strike_monot_violation' column for one option_type ('C' or 'P').

    Within each (trade_date, expiry_date), sorted by strike:
      - Calls: price must be non-increasing in strike.
      - Puts : price must be non-decreasing in strike.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    df_sub = (
        df_long
        .filter(pl.col("option_type") == option_type)
        .sort(["trade_date", "expiry_date", "strike"])
        .with_columns(
            # forward difference: price(K+1) - price(K)
            forward_diff = pl.col(price_col).shift(-1) - pl.col(price_col)
        )
        .with_columns(
            strike_monot_violation = pl.when(option_type == "C")
            # Calls: C(K+1) <= C(K)  ⇒ diff <= 0  (violation if > tol)
            .then(pl.col("forward_diff") > tol)
            # Puts : P(K+1) >= P(K)  ⇒ diff >= 0  (violation if < -tol)
            .otherwise(pl.col("forward_diff") < -tol)
            .fill_null(False)  # last row in each group has diff = null
        )
        .drop("forward_diff")
    )

    return df_sub


def summarize_monotonicity_by_region(
    df: pl.DataFrame,
    *,
    violation_col: str = "strike_monot_violation",
    dte_col: str = "dte",
    delta_col: str = "delta",
    dte_bins: Sequence[int | float] = (0, 10, 30, 60),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> pl.DataFrame:
    """
    Summarise monotonicity violations by (DTE, delta) region.

    Parameters
    ----------
    df :
        Long ORATS DataFrame (after flagging monotonicity).
    violation_col :
        Name of the boolean column indicating a violation.
    dte_col :
        Column with days-to-expiry (default: "dte").
    delta_col :
        Column with delta (default: "delta").
    dte_bins :
        Bin edges for DTE bucketing.
    delta_bins :
        Bin edges for delta bucketing.

    Returns
    -------
    pl.DataFrame
        One row per (dte_bucket, delta_bucket) with:
        - n_viol     : number of violations in bucket
        - n_rows     : total rows in bucket
        - viol_rate  : n_viol / n_rows       (within-bucket rate)
        - viol_share : n_viol / total_viol   (share of all violations)
        - row_share  : n_rows / total_rows   (share of all rows)
    """
    totals = df.select(
        pl.col(violation_col).sum().alias("total_viol"),
        pl.len().alias("total_rows"),
    ).row(0)
    total_viol, total_rows = totals

    summary = (
        df
        .group_by(
            pl.col(dte_col).cut(dte_bins).alias("dte_bucket"),
            pl.col(delta_col).cut(delta_bins).alias("delta_bucket"),
        )
        .agg(
            pl.col(violation_col).sum().alias("n_viol"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("n_viol") / pl.col("n_rows")).alias("viol_rate"),
            (pl.col("n_viol") / total_viol).alias("viol_share"),
            (pl.col("n_rows") / total_rows).alias("row_share"),
        )
        .sort("viol_rate", descending=True)
    )

    return summary