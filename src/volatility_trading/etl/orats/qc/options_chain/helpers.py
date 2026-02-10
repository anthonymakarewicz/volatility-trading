"""Helper utilities for options-chain QC runner orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from volatility_trading.datasets import (
    options_chain_path,
    options_chain_wide_to_long,
    scan_options_chain,
)


def get_parquet_path(proc_root: Path, ticker: str) -> Path | None:
    """Best-effort locate the processed options-chain parquet path."""
    try:
        return options_chain_path(proc_root, ticker)
    except Exception:
        return None


def read_exercise_style(*, parquet_path: Path | None) -> str | None:
    """Best-effort read `exercise_style` (`EU`/`AM`) from `manifest.json`."""
    if parquet_path is None:
        return None

    try:
        manifest_path = parquet_path.parent / "manifest.json"
        if not manifest_path.exists():
            return None

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        params = payload.get("params")
        if not isinstance(params, dict):
            return None

        style = params.get("exercise_style", None)
        return style if style in {"EU", "AM"} else None
    except Exception:
        return None


def load_options_chain_df(*, ticker: str, proc_root: Path) -> pl.DataFrame:
    """Load processed options chain as a long Polars DataFrame."""
    lf = scan_options_chain(ticker, proc_root=proc_root)
    return options_chain_wide_to_long(lf).collect()


def apply_roi_filter(
    df: pl.DataFrame,
    *,
    dte_min: int = 10,
    dte_max: int = 60,
    delta_min: float = 0.1,
    delta_max: float = 0.9,
) -> pl.DataFrame:
    """Filter to the ROI subset used by SOFT/INFO reporting."""
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )
