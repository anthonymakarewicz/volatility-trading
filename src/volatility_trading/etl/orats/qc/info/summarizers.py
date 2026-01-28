# qc/info/summarizers.py
from __future__ import annotations

from typing import Any

import polars as pl


def summarize_volume_oi_metrics(
    *,
    df: pl.DataFrame,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> dict[str, Any]:
    """
    Summarize volume / open interest metrics.

    NOTE: Keep logic identical to your previous `checks_info.py`.
    If you already had a specific payload/keys, paste that code here as-is.
    """
    if df.height == 0:
        return {"n_rows": 0}

    out: dict[str, Any] = {"n_rows": int(df.height)}

    if volume_col in df.columns:
        s = df.get_column(volume_col)
        out.update(
            {
                "volume_null_rate": float(s.is_null().mean()),
                "volume_zero_rate": float((s.fill_null(0) == 0).mean()),
            }
        )

    if oi_col in df.columns:
        s = df.get_column(oi_col)
        out.update(
            {
                "oi_null_rate": float(s.is_null().mean()),
                "oi_zero_rate": float((s.fill_null(0) == 0).mean()),
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

    NOTE: Keep logic identical to your previous `checks_info.py`.
    If you already had specific quantiles/keys, paste that code here as-is.
    """
    if df.height == 0:
        return {"n_rows": 0}

    out: dict[str, Any] = {"n_rows": int(df.height)}

    if r_col not in df.columns:
        out["reason"] = f"missing col {r_col!r}"
        return out

    s = df.get_column(r_col)
    s_nonnull = s.drop_nulls()
    out["r_null_rate"] = float(s.is_null().mean())

    if len(s_nonnull) == 0:
        out["reason"] = "all risk_free_rate null"
        return out

    out.update(
        {
            "r_min": float(s_nonnull.min()),
            "r_max": float(s_nonnull.max()),
            "r_mean": float(s_nonnull.mean()),
            "r_median": float(s_nonnull.median()),
        }
    )

    return out