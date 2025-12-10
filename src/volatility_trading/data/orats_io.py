"""
Common data utilities used for input/output operations
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import polars as pl

from volatility_trading.config.paths import PROC_ORATS


def load_orats_panel_lazy(
    ticker: str,
    proc_root: Path | str = PROC_ORATS,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """
    Return a LazyFrame over the processed WIDE ORATS panel for a ticker.
    """
    proc_root = Path(proc_root)
    path = proc_root / f"orats_panel_{ticker}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Processed ORATS panel not found: {path}")

    lf = pl.scan_parquet(path)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def load_orats_panel(
    ticker: str,
    proc_root: Path | str = PROC_ORATS,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Eager version: loads the processed WIDE ORATS panel for a ticker.
    """
    return load_orats_panel_lazy(ticker, proc_root, columns).collect()


def orats_wide_to_long(wide: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a WIDE ORATS panel (call_* / put_*) to LONG with option_type ∈ {"C","P"}.
    """
    call_cols = [c for c in wide.columns if c.startswith("call_")]
    put_cols  = [c for c in wide.columns if c.startswith("put_")]

    # base columns common to both
    base_cols = [c for c in wide.columns if not c.startswith(("call_", "put_"))]

    calls = (
        wide
        .select(base_cols + call_cols)
        .rename({c: c.removeprefix("call_") for c in call_cols})
        .with_columns(pl.lit("C").alias("option_type"))
    )

    puts = (
        wide
        .select(base_cols + put_cols)
        .rename({c: c.removeprefix("put_") for c in put_cols})
        .with_columns(pl.lit("P").alias("option_type"))
    )

    return pl.concat([calls, puts], how="vertical")