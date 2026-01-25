# soft/utils.py
from __future__ import annotations

import polars as pl
from volatility_trading.datasets import options_chain_long_to_wide


def _iter_subsets_for_spec(
    *,
    spec: dict,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    df_wide_global: pl.DataFrame | None,
    df_wide_roi: pl.DataFrame | None,
) -> list[tuple[str, pl.DataFrame]]:
    requires_wide = bool(spec.get("requires_wide", False))
    use_roi = bool(spec.get("use_roi", True))

    if requires_wide:
        if df_wide_global is None:
            return []
        out: list[tuple[str, pl.DataFrame]] = [("GLOBAL", df_wide_global)]
        if use_roi and df_wide_roi is not None:
            out.append(("ROI", df_wide_roi))
        return out

    out = [("GLOBAL", df_global)]
    if use_roi:
        out.append(("ROI", df_roi))
    return out


def _build_wide_views_if_needed(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    soft_specs: list[dict],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    def build_wide(df_long: pl.DataFrame) -> pl.DataFrame:
        wide = (
            options_chain_long_to_wide(long=df_long, how="inner")
            .collect()
        )
        if "call_delta" in wide.columns:
            wide = wide.with_columns(pl.col("call_delta").alias("delta"))
        return wide

    needs_wide = any(bool(spec.get("requires_wide", False)) for spec in soft_specs)
    if not needs_wide:
        return None, None

    return build_wide(df_global), build_wide(df_roi)