# qc/soft/dataset_checks/risk_free_rate.py
from __future__ import annotations

from typing import Any

import polars as pl

from ...serialization import df_to_jsonable_records


def check_unique_rf_rate_per_day_expiry(
    *,
    df: pl.DataFrame,
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
    r_col: str = "risk_free_rate",
    tol_abs: float = 1e-4,  # ~1bp absolute tolerance
    tol_rel: float = 0.0,  # optional relative tolerance
    rel_floor: float = 1e-6,  # avoid div by ~0
    max_examples: int = 5,
) -> dict[str, Any]:
    """
    Unique risk-free rate per (trade_date, expiry_date) sanity check.

    Returns (required keys):
      - viol_rate: float  (#viol_units / #units)
      - n_viol: int       (#(trade_date, expiry) groups violating)
      - n_units: int      (#(trade_date, expiry) groups examined)
    """
    required_cols = {trade_col, expiry_col, r_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # Only evaluate where r is present
    dfx = df.select([trade_col, expiry_col, r_col]).filter(pl.col(r_col).is_not_null())
    if dfx.height == 0:
        return {
            "viol_rate": 0.0,
            "n_viol": 0,
            "n_units": 0,
            "reason": "no non-null risk_free_rate rows",
        }

    # Group by (trade_date, expiry_date) and measure spread
    g = [trade_col, expiry_col]
    grp = (
        dfx.group_by(g)
        .agg(
            pl.col(r_col).min().alias("r_min"),
            pl.col(r_col).max().alias("r_max"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("r_max") - pl.col("r_min")).alias("r_spread"),
        )
    )

    # Tolerance: abs OR (rel * max(|rate|, floor))
    abs_bad = pl.col("r_spread") > tol_abs
    rel_bad = pl.col("r_spread") > (
        tol_rel * pl.max_horizontal([pl.col("r_max").abs(), pl.lit(rel_floor)])
    )
    viol_expr = abs_bad | rel_bad

    grp2 = grp.with_columns(viol_expr.alias("viol"))

    n_units = int(grp2.height)
    n_viol = int(grp2.select(pl.col("viol").sum().alias("n"))["n"][0])

    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    # Some helpful debug payload: worst offenders
    examples_df = (
        grp2.filter(pl.col("viol")).sort("r_spread", descending=True).head(max_examples)
    )

    max_spread = float(grp2.select(pl.col("r_spread").max().alias("m"))["m"][0])

    return {
        "viol_rate": float(viol_rate),
        "n_viol": n_viol,
        "n_units": n_units,
        "tol_abs": float(tol_abs),
        "tol_rel": float(tol_rel),
        "max_spread": max_spread,
        "examples": df_to_jsonable_records(examples_df),
    }
