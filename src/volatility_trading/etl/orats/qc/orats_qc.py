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
    """Flag strike monotonicity violations for one option type ('C' or 'P').

    Within each (trade_date, expiry_date), sorted by strike:
      - Calls: price must be non-increasing in strike.
      - Puts : price must be non-decreasing in strike.

    Notes
    -----
    The forward difference must be computed *within* each (trade_date, expiry_date)
    group. Without `.over(["trade_date", "expiry_date"])`, `shift(-1)` would compare
    the last strike of one expiry to the first strike of the next expiry.

    Returns
    -------
    pl.DataFrame
        Subset for the requested option_type, with:
        - `monot_violation` (bool): strike monotonicity violation flag
        - `strike_monot_violation` (bool): alias of `monot_violation` for clarity
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    df_sub = (
        df_long
        .filter(pl.col("option_type") == option_type)
        .sort(["trade_date", "expiry_date", "strike"])
        .with_columns(
            # forward difference: price(K_next) - price(K), computed within group
            forward_diff=(
                pl.col(price_col).shift(-1).over(["trade_date", "expiry_date"])
                - pl.col(price_col)
            )
        )
        .with_columns(
            monot_violation=(
                pl.when(option_type == "C")
                # Calls: C(K_next) <= C(K)  ⇒ diff <= 0  (violation if > tol)
                .then(pl.col("forward_diff") > tol)
                # Puts : P(K_next) >= P(K)  ⇒ diff >= 0  (violation if < -tol)
                .otherwise(pl.col("forward_diff") < -tol)
                .fill_null(False)  # last row in each group has diff = null
            )
        )
        .drop("forward_diff")
    )

    return df_sub


def flag_maturity_monotonicity_long(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    price_col: str = "mid_price",
    tol: float = 1e-6,
    expiry_col: str = "expiry_date",
) -> pl.DataFrame:
    """Flag maturity monotonicity violations (calendar arbitrage) for one option type.

    For European options (and in arbitrage-free surfaces), option value should be
    non-decreasing with time-to-expiry for fixed (trade_date, strike).

    Within each (trade_date, strike), sorted by expiry:
      - Calls:  C(T2) >= C(T1) for T2 > T1
      - Puts :  P(T2) >= P(T1) for T2 > T1

    We flag a violation when the *forward* difference is negative beyond `tol`:
        price(T_next) - price(T) < -tol

    Parameters
    ----------
    df_long:
        Long ORATS DataFrame with at least:
        ["trade_date", "option_type", "strike", expiry_col, price_col].
    option_type:
        'C' for calls or 'P' for puts.
    price_col:
        Column to test for monotonicity (default: "mid_price").
    tol:
        Tolerance for numerical noise.
    expiry_col:
        Column to use for sorting maturities (default: "expiry_date").

    Returns
    -------
    pl.DataFrame
        Subset for the requested option_type, with an added boolean column
        `maturity_monot_violation`.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    # We compute the forward difference within each (trade_date, strike).
    # Note: shift(-1) is applied in row order *after* sorting.
    df_sub = (
        df_long
        .filter(pl.col("option_type") == option_type)
        .sort(["trade_date", "strike", expiry_col])
        .with_columns(
            forward_diff=(
                pl.col(price_col).shift(-1).over(["trade_date", "strike"])
                - pl.col(price_col)
            )
        )
        .with_columns(
            monot_violation=(pl.col("forward_diff") < -tol).fill_null(False)
        )
        .drop("forward_diff")
    )

    return df_sub


def summarize_monotonicity_by_region(
    df: pl.DataFrame,
    *,
    violation_col: str = "monot_violation",
    dte_col: str = "dte",
    delta_col: str = "delta",
    dte_bins: Sequence[int | float] = (0, 10, 30, 60),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> pl.DataFrame:
    """Summarise monotonicity violations by (DTE, delta) region.

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
            pl.col(delta_col).abs().cut(delta_bins).alias("delta_bucket"),
        )
        .agg(
            pl.col(violation_col).sum().alias("n_viol"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("n_viol") / pl.col("n_rows")).alias("viol_rate_bucket"),
            (pl.col("n_viol") / total_viol).alias("viol_share"),
            (pl.col("n_viol") / total_rows).alias("viol_rate_glob"),
        )
        .sort("viol_rate_bucket", descending=True)
    )

    return summary