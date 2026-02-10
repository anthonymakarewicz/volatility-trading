"""Informational metric summarizers for QC reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def _to_float(value: Any) -> float:
    """Convert a Polars scalar summary value to float."""
    return float(value)


def summarize_volume_oi_metrics(
    *,
    df: pl.DataFrame,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> dict[str, Any]:
    """
    Summarize volume / open interest metrics.
    """
    if df.height == 0:
        return {"n_rows": 0}

    out: dict[str, Any] = {"n_rows": int(df.height)}

    if volume_col in df.columns:
        s = df.get_column(volume_col)
        out.update(
            {
                "volume_null_rate": _to_float(s.is_null().mean()),
                "volume_zero_rate": _to_float((s.fill_null(0) == 0).mean()),
            }
        )

    if oi_col in df.columns:
        s = df.get_column(oi_col)
        out.update(
            {
                "oi_null_rate": _to_float(s.is_null().mean()),
                "oi_zero_rate": _to_float((s.fill_null(0) == 0).mean()),
            }
        )

    return out


def summarize_risk_free_rate_metrics(
    *,
    df: pl.DataFrame,
    r_col: str = "risk_free_rate",
) -> dict[str, Any]:
    """
    Summarize risk-free rate metrics.
    """
    if df.height == 0:
        return {"n_rows": 0}

    out: dict[str, Any] = {"n_rows": int(df.height)}

    if r_col not in df.columns:
        out["reason"] = f"missing col {r_col!r}"
        return out

    s = df.get_column(r_col)
    s_nonnull = s.drop_nulls()
    out["r_null_rate"] = _to_float(s.is_null().mean())

    if len(s_nonnull) == 0:
        out["reason"] = "all risk_free_rate null"
        return out

    out.update(
        {
            "r_min": _to_float(s_nonnull.min()),
            "r_max": _to_float(s_nonnull.max()),
            "r_mean": _to_float(s_nonnull.mean()),
            "r_median": _to_float(s_nonnull.median()),
        }
    )

    return out


def summarize_core_numeric_stats(
    *,
    df: pl.DataFrame,
    cols: Sequence[str],
    quantiles: tuple[float, ...] = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0),
    strict: bool = False,
) -> dict[str, Any]:
    """
    Summarize basic numeric stats for a provided set of columns.
    """
    if df.height == 0:
        return {"n_rows": 0, "cols_used": [], "missing_cols": [], "stats": {}}

    use_cols = list(cols)

    missing_cols: list[str] = [c for c in use_cols if c not in df.columns]
    present_cols: list[str] = [c for c in use_cols if c in df.columns]

    if strict and missing_cols:
        # In INFO checks we usually don't want to raise, but strict is there
        raise ValueError(f"missing columns: {missing_cols}")

    stats: dict[str, Any] = {}

    for c in present_cols:
        s = df.get_column(c)

        # Only numeric dtypes
        if not (s.dtype.is_numeric() or s.dtype == pl.Decimal):
            if strict:
                missing_cols.append(c)
            continue

        # Decimal -> Float64 for easier summary/JSON
        if s.dtype == pl.Decimal:
            s = s.cast(pl.Float64)

        null_rate = _to_float(s.is_null().mean())
        s_nonnull = s.drop_nulls()
        n_nonnull = len(s_nonnull)

        if n_nonnull == 0:
            stats[c] = {
                "null_rate": null_rate,
                "n_nonnull": 0,
                "reason": "all null",
            }
            continue

        # Base stats
        col_stats: dict[str, Any] = {
            "null_rate": null_rate,
            "n_nonnull": n_nonnull,
            "min": _to_float(s_nonnull.min()),
            "max": _to_float(s_nonnull.max()),
            "mean": _to_float(s_nonnull.mean()),
            "std": _to_float(s_nonnull.std()),
            "median": _to_float(s_nonnull.median()),
        }

        # Quantiles
        for q in quantiles:
            # Polars returns scalar; for safety cast to float
            v = s_nonnull.quantile(q, "nearest")
            key = f"q_{q:.2f}"
            col_stats[key] = _to_float(v)

        stats[c] = col_stats

    return {
        "n_rows": int(df.height),
        "cols_used": use_cols,
        "missing_cols": missing_cols,
        "stats": stats,
    }
