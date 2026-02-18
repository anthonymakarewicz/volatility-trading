"""I/O helpers for the processed ORATS daily-features panel.

Provides a thin, opinionated interface to:
- resolve ticker-level parquet paths
- scan data lazily for query composition
- read data eagerly for in-memory use
- join daily-features columns onto existing keyed panels
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from volatility_trading.config.paths import PROC_ORATS_DAILY_FEATURES

JoinStrategy = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]

DEFAULT_BASE_COLS = [
    "ticker",
    "trade_date",
]

# TODO(API): rename path helper functions to use `get_*` prefix
# for consistency across options_chain and daily_features modules.


def daily_features_path(proc_root: Path | str, ticker: str) -> Path:
    """Build the processed daily-features parquet path for one ticker.

    Args:
        proc_root: Root directory of the processed dataset.
        ticker: Underlying symbol (for example `SPX` or `AAPL`).

    Returns:
        Filesystem path to `underlying=<TICKER>/part-0000.parquet`.

    Raises:
        ValueError: If `ticker` is empty after normalization.
    """
    root = Path(proc_root)
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("ticker must be non-empty")
    return root / f"underlying={t}" / "part-0000.parquet"


def scan_daily_features(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_DAILY_FEATURES,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan the processed daily-features parquet lazily.

    Args:
        ticker: Underlying symbol to load.
        proc_root: Root directory of the processed daily-features dataset.
        columns: Optional projection list to select while scanning.

    Returns:
        Lazy frame backed by `underlying=<TICKER>/part-0000.parquet`.

    Raises:
        FileNotFoundError: If the ticker parquet file does not exist.
    """
    root = Path(proc_root)
    path = daily_features_path(root, ticker)

    if not path.exists():
        raise FileNotFoundError(f"Processed daily features not found: {path}")

    lf = pl.scan_parquet(path)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_daily_features(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_DAILY_FEATURES,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read the processed daily-features parquet eagerly.

    Args:
        ticker: Underlying symbol to load.
        proc_root: Root directory of the processed daily-features dataset.
        columns: Optional projection list to load.

    Returns:
        DataFrame keyed by `(ticker, trade_date)`.
    """
    return scan_daily_features(
        ticker,
        proc_root=proc_root,
        columns=columns,
    ).collect()


def join_daily_features(
    lf: pl.LazyFrame,
    *,
    ticker: str,
    proc_root: Path | str = PROC_ORATS_DAILY_FEATURES,
    columns: Sequence[str] | None = None,
    how: JoinStrategy = "left",
) -> pl.LazyFrame:
    """Join ticker daily-features columns onto an existing lazy panel.

    Assumes `lf` includes join keys `(ticker, trade_date)`.

    Args:
        lf: Base lazy frame that already contains key columns.
        ticker: Ticker whose daily-features panel should be joined.
        proc_root: Root directory of the processed daily-features dataset.
        columns: Optional subset of feature columns to include.
        how: Join strategy passed to `LazyFrame.join`.

    Returns:
        Base frame with daily-features columns appended.
    """
    features = scan_daily_features(ticker, proc_root=proc_root, columns=columns)

    # Avoid duplicating keys if user selected them in `columns`.
    if columns is not None:
        schema = features.collect_schema()
        cols = schema.names()
        drop_keys = [c for c in DEFAULT_BASE_COLS if c in cols]
        if drop_keys:
            features = features.drop(drop_keys)

    return lf.join(features, on=DEFAULT_BASE_COLS, how=how)
