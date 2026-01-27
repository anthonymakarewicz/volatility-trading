# qc/soft/dataset_checks/spot_forward.py
from __future__ import annotations

from typing import Any

import polars as pl

from .utils import _make_examples_json_safe


def check_spot_constant_per_trade_date(
    *,
    df: pl.DataFrame,
    trade_col: str = "trade_date",
    spot_col: str = "spot_price",
    tol_abs: float = 0.001,     # absolute tolerance in spot units
    tol_rel: float = 5e-4,      # 5 bps relative tolerance
    rel_floor: float = 1e-6,
    max_examples: int = 20,
) -> dict[str, Any]:
    """
    Spot should be (nearly) constant within each trade_date.
    """
    required = {trade_col, spot_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    dfx = (
        df
        .select([trade_col, spot_col])
        .filter(pl.col(spot_col).is_not_null())
    )
    if dfx.height == 0:
        return {
            "viol_rate": 0.0,
            "n_viol": 0,
            "n_units": 0,
            "reason": "no non-null spot rows",
        }

    grp = (
        dfx.group_by([trade_col])
        .agg(
            pl.col(spot_col).min().alias("spot_min"),
            pl.col(spot_col).max().alias("spot_max"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("spot_max") - pl.col("spot_min")).alias("spot_spread")
        )
    )

    abs_bad = pl.col("spot_spread") > tol_abs
    rel_bad = (
        pl.col("spot_spread")
        > (
            tol_rel
            * pl.max_horizontal(
                [pl.col("spot_max").abs(), pl.lit(rel_floor)]
            )
        )
    )
    viol = (abs_bad | rel_bad).alias("viol")

    grp2 = grp.with_columns(viol)

    n_units = int(grp2.height)
    n_viol = int(grp2.select(pl.col("viol").sum().alias("n"))["n"][0])
    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    examples_df = (
        grp2.filter(pl.col("viol"))
        .sort("spot_spread", descending=True)
        .head(max_examples)
    )

    max_spread = float(
        grp2.select(pl.col("spot_spread").max().alias("m"))["m"][0]
    )

    return {
        "viol_rate": float(viol_rate),
        "n_viol": n_viol,
        "n_units": n_units,
        "tol_abs": float(tol_abs),
        "tol_rel": float(tol_rel),
        "max_spread": max_spread,
        "examples": _make_examples_json_safe(examples_df),
    }


def check_forward_constant_per_trade_date_expiry(
    *,
    df: pl.DataFrame,
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
    fwd_col: str = "underlying_price",  # ORATS implied forward for EU
    tol_abs: float = 0.001,
    tol_rel: float = 5e-4,
    rel_floor: float = 1e-6,
    max_examples: int = 20,
) -> dict[str, Any]:
    """
    EU: implied forward should be (nearly) constant within each
    (trade_date, expiry_date), i.e. same across strikes and option_type
    for that slice.
    """
    required = {trade_col, expiry_col, fwd_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    dfx = (
        df
        .select([trade_col, expiry_col, fwd_col])
        .filter(pl.col(fwd_col).is_not_null())
    )
    if dfx.height == 0:
        return {
            "viol_rate": 0.0,
            "n_viol": 0,
            "n_units": 0,
            "reason": "no non-null forward rows",
        }

    g = [trade_col, expiry_col]
    grp = (
        dfx.group_by(g)
        .agg(
            pl.col(fwd_col).min().alias("fwd_min"),
            pl.col(fwd_col).max().alias("fwd_max"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("fwd_max") - pl.col("fwd_min")).alias("fwd_spread")
        )
    )

    abs_bad = pl.col("fwd_spread") > tol_abs
    rel_bad = (
        pl.col("fwd_spread")
        > (
            tol_rel
            * pl.max_horizontal(
                [pl.col("fwd_max").abs(), pl.lit(rel_floor)]
            )
        )
    )
    grp2 = grp.with_columns((abs_bad | rel_bad).alias("viol"))

    n_units = int(grp2.height)
    n_viol = int(grp2.select(pl.col("viol").sum().alias("n"))["n"][0])
    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    examples_df = (
        grp2.filter(pl.col("viol"))
        .sort("fwd_spread", descending=True)
        .head(max_examples)
    )

    max_spread = float(
        grp2.select(pl.col("fwd_spread").max().alias("m"))["m"][0]
    )

    return {
        "viol_rate": float(viol_rate),
        "n_viol": n_viol,
        "n_units": n_units,
        "tol_abs": float(tol_abs),
        "tol_rel": float(tol_rel),
        "max_spread": max_spread,
        "examples": _make_examples_json_safe(examples_df),
    }


def check_spot_equals_underlying_per_trade_date_am(
    *,
    df: pl.DataFrame,
    trade_col: str = "trade_date",
    spot_col: str = "spot_price",
    underlying_col: str = "underlying_price",
    tol_abs: float = 0.001,
    tol_rel: float = 5e-4,
    rel_floor: float = 1e-6,
    max_examples: int = 20,
) -> dict[str, Any]:
    """
    AM: underlying_price should be the spot (same snapshot),
    so within each trade_date max(|spot - underlying|) should be tiny.
    """
    required = {trade_col, spot_col, underlying_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    dfx = (
        df
        .select([trade_col, spot_col, underlying_col])
        .filter(
            pl.col(spot_col).is_not_null()
            & pl.col(underlying_col).is_not_null()
        )
    )
    if dfx.height == 0:
        return {
            "viol_rate": 0.0,
            "n_viol": 0,
            "n_units": 0,
            "reason": "no rows where both spot and underlying are non-null",
        }

    grp = (
        dfx.with_columns(
            (pl.col(spot_col) - pl.col(underlying_col)).abs().alias(
                "abs_diff"
            )
        )
        .group_by([trade_col])
        .agg(
            pl.col("abs_diff").max().alias("max_abs_diff"),
            pl.col(spot_col).max().alias("spot_ref"),
            pl.len().alias("n_rows"),
        )
    )

    abs_bad = pl.col("max_abs_diff") > tol_abs
    rel_bad = (
        pl.col("max_abs_diff")
        > (
            tol_rel
            * pl.max_horizontal(
                [pl.col("spot_ref").abs(), pl.lit(rel_floor)]
            )
        )
    )

    grp2 = grp.with_columns((abs_bad | rel_bad).alias("viol"))

    n_units = int(grp2.height)
    n_viol = int(grp2.select(pl.col("viol").sum().alias("n"))["n"][0])
    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    examples_df = (
        grp2.filter(pl.col("viol"))
        .sort("max_abs_diff", descending=True)
        .head(max_examples)
    )

    max_diff = float(
        grp2.select(pl.col("max_abs_diff").max().alias("m"))["m"][0]
    )

    return {
        "viol_rate": float(viol_rate),
        "n_viol": n_viol,
        "n_units": n_units,
        "tol_abs": float(tol_abs),
        "tol_rel": float(tol_rel),
        "max_abs_diff": max_diff,
        "examples": _make_examples_json_safe(examples_df),
    }