from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def summarize_by_bucket(
    df: pl.DataFrame,
    *,
    violation_col: str,
    dte_col: str = "dte",
    delta_col: str = "delta",
    dte_bins: Sequence[int | float] = (0, 10, 30, 60),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    min_rows: int = 50,
    top_k: int | None = 10,
) -> pl.DataFrame:
    """Summarise violations by (DTE bucket, |delta| bucket)."""
    if df.height == 0 or violation_col not in df.columns:
        return pl.DataFrame()

    totals = df.select(
        pl.col(violation_col).sum().alias("total_viol"),
        pl.len().alias("total_rows"),
    ).row(0)

    total_viol, total_rows = totals
    total_viol = int(total_viol or 0)

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
            pl.when(pl.lit(total_viol) > 0)
            .then(pl.col("n_viol") / pl.lit(total_viol))
            .otherwise(pl.lit(0.0))
            .alias("viol_share"),
            (pl.col("n_rows") / pl.lit(total_rows)).alias("row_share"),
        )
        .filter(pl.col("n_rows") >= min_rows)
        .sort(["viol_rate_bucket", "n_viol"], descending=True)
    )

    if top_k is not None:
        summary = summary.head(top_k)

    return summary