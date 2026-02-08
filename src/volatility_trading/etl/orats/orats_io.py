"""
This module provides a small, opinionated I/O layer around the processed
ORATS panels produced by the ETL pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_ORATS


def load_orats_panel_lazy(
    ticker: str,
    proc_root: Path | str = PROC_ORATS,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Return a lazy view over the processed WIDE ORATS panel for a ticker.

    Parameters
    ----------
    ticker : str
        Underlying symbol (e.g. "SPX"). Used to locate the corresponding
        processed Parquet file named ``orats_panel_{ticker}.parquet`` under
        ``proc_root``.
    proc_root : Path | str, default=PROC_ORATS
        Root directory where processed ORATS panels are stored. By default
        this points to the project-level ``data/processed/orats`` folder.
    columns : Sequence[str] | None, optional
        Optional list of column names to project. If provided, only these
        columns are included in the returned LazyFrame.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame over the WIDE panel. No data is loaded into
        memory until a terminal operation (e.g. ``collect()``) is called.

    Raises
    ------
    FileNotFoundError
        If the expected Parquet file for the ticker does not exist.
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
    """Eagerly load the processed WIDE ORATS panel for a ticker.

    This is a convenience wrapper around ``load_orats_panel_lazy`` that
    immediately materialises the data into a Polars DataFrame.

    Parameters
    ----------
    ticker : str
        Underlying symbol (e.g. "SPX").
    proc_root : Path | str, default=PROC_ORATS
        Root directory where processed ORATS panels are stored.
    columns : Sequence[str] | None, optional
        Optional list of column names to project before collecting.

    Returns
    -------
    pl.DataFrame
        In-memory Polars DataFrame containing the WIDE panel for the
        requested ticker.

    Raises
    ------
    FileNotFoundError
        If the expected Parquet file for the ticker does not exist.
    """
    return load_orats_panel_lazy(ticker, proc_root, columns).collect()


def orats_wide_to_long(wide: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Convert a WIDE ORATS panel into a LONG panel with option_type.

    The input is expected to follow the processed ORATS WIDE schema, where
    call-specific fields are prefixed with ``call_`` and put-specific fields
    with ``put_`` (e.g. ``call_bid``, ``put_bid``, ``call_delta``,
    ``put_delta``). Columns not starting with ``call_`` or ``put_`` are
    assumed to be common to both legs.

    The function returns a LONG panel in which each physical option contract
    (call or put) occupies one row, differentiated by an ``option_type``
    column equal to ``"C"`` for calls and ``"P"`` for puts.

    Parameters
    ----------
    wide : pl.DataFrame
        Processed WIDE ORATS panel for a single underlying, with both
        call_* and put_* columns.

    Returns
    -------
    pl.DataFrame
        LONG panel with an ``option_type`` column and all call_/put_ columns
        de-prefixed so that calls and puts share a common set of column
        names (e.g. ``bid``, ``ask``, ``delta``).
    """
    # Normalise to LazyFrame
    lf = wide.lazy() if isinstance(wide, pl.DataFrame) else wide

    # Resolve schema *without* materialising data
    schema = lf.collect_schema()
    cols = list(schema.names())

    call_cols = [c for c in cols if c.startswith("call_")]
    put_cols = [c for c in cols if c.startswith("put_")]
    base_cols = [c for c in cols if not c.startswith(("call_", "put_"))]

    if not call_cols and not put_cols:
        raise ValueError(
            "orats_wide_to_long expected at least one 'call_*' or 'put_*' "
            "column, but found none."
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
    long = long.with_columns(pl.col("option_type").cast(pl.Categorical))

    return long
