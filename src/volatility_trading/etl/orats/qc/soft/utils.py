# qc/soft/utils.py
from __future__ import annotations

import polars as pl

from volatility_trading.datasets import options_chain_long_to_wide

from ..spec_types import SoftSpec


def _iter_subsets_for_spec(
    *,
    spec: SoftSpec,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    df_wide_global: pl.DataFrame | None,
    df_wide_roi: pl.DataFrame | None,
) -> list[tuple[str, pl.DataFrame]]:
    """
    Return the list of (label, df) subsets to run for this spec,
    depending on requires_wide and use_roi.
    """
    # Only row specs can require WIDE
    if spec.kind == "row" and spec.requires_wide:
        if df_wide_global is None:
            return []
        out: list[tuple[str, pl.DataFrame]] = [("GLOBAL", df_wide_global)]
        if spec.use_roi and df_wide_roi is not None:
            out.append(("ROI", df_wide_roi))
        return out

    out: list[tuple[str, pl.DataFrame]] = [("GLOBAL", df_global)]
    if spec.use_roi:
        out.append(("ROI", df_roi))
    return out


def _build_wide_views_if_needed(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    soft_specs: list[SoftSpec],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """
    Build strict paired WIDE views only if at least one row-spec requires_wide=True.
    """

    def _build_wide(df_long: pl.DataFrame) -> pl.DataFrame:
        wide = (
            options_chain_long_to_wide(long=df_long, how="inner")
            .collect()
        )
        if "call_delta" in wide.columns:
            wide = wide.with_columns(pl.col("call_delta").alias("delta"))
        return wide

    # Only row specs can set requires_wide
    needs_wide = any(
        (spec.kind == "row" and spec.requires_wide)
        for spec in soft_specs
    )
    if not needs_wide:
        return None, None

    return _build_wide(df_global), _build_wide(df_roi)