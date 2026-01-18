"""
Small, opinionated I/O layer for the processed options chain dataset.

Conventions
----------
- scan_* returns a Polars LazyFrame
- read_* returns a Polars DataFrame
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN


def options_chain_path(proc_root: Path | str, ticker: str) -> Path:
    """
    Return the processed options chain parquet path for a ticker.
    """
    root = Path(proc_root)
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("ticker must be non-empty")
    return root / f"underlying={t}" / "part-0000.parquet"


def scan_options_chain(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_OPTIONS_CHAIN,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Lazy scan of the processed WIDE options chain for one ticker."""
    root = Path(proc_root)
    path = options_chain_path(root, ticker)

    if not path.exists():
        raise FileNotFoundError(f"Processed options chain not found: {path}")

    lf = pl.scan_parquet(path)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_options_chain(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_OPTIONS_CHAIN,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Eager read of the processed WIDE options chain for one ticker."""
    return (scan_options_chain(
        ticker, 
        proc_root=proc_root, 
        columns=columns).collect()
    )


def options_chain_wide_to_long(
    wide: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame:
    """Convert a WIDE options chain into a LONG chain with option_type.

    Expects call_* and/or put_* prefixed columns. Non call_/put_ columns are
    treated as shared (base) columns.
    """
    lf = wide.lazy() if isinstance(wide, pl.DataFrame) else wide

    schema = lf.collect_schema()
    cols = list(schema.names())

    call_cols = [c for c in cols if c.startswith("call_")]
    put_cols = [c for c in cols if c.startswith("put_")]
    base_cols = [c for c in cols if not c.startswith(("call_", "put_"))]

    if not call_cols and not put_cols:
        raise ValueError(
            "options_chain_wide_to_long expected at least one "
            "'call_*' or 'put_*' column, but found none."
        )

    calls = (
        lf.select(base_cols + call_cols)
        .rename({c: c.removeprefix("call_") for c in call_cols})
        .with_columns(pl.lit("C").alias("option_type"))
    )

    puts = (
        lf.select(base_cols + put_cols)
        .rename({c: c.removeprefix("put_") for c in put_cols})
        .with_columns(pl.lit("P").alias("option_type"))
    )

    long = pl.concat([calls, puts], how="vertical")
    return long.with_columns(pl.col("option_type").cast(pl.Categorical))