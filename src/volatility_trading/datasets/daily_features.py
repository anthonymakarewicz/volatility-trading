"""
volatility_trading.datasets.daily_features
------------------------------------------
Small, opinionated I/O layer for the processed ORATS daily-features dataset.

Conventions
----------
- scan_* returns a Polars LazyFrame
- read_* returns a Polars DataFrame
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_ORATS_DAILY_FEATURES


DEFAULT_BASE_COLS = [
    "ticker",
    "trade_date",
]


def daily_features_path(proc_root: Path | str, ticker: str) -> Path:
    """
    Return the processed daily-features parquet path for a ticker.
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
    """Scan the processed ORATS daily-features panel for one ticker (lazy).

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. "SPX", "SPY").
    proc_root:
        Root directory of the processed daily-features dataset.
    columns:
        Optional column subset to project during the scan.

    Returns
    -------
    pl.LazyFrame
        LazyFrame pointing to `underlying=<TICKER>/part-0000.parquet`.

    Raises
    ------
    FileNotFoundError
        If the processed parquet does not exist.
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
    """Read the processed ORATS daily-features panel for one ticker (eager).

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. "SPX", "SPY").
    proc_root:
        Root directory of the processed daily-features dataset.
    columns:
        Optional list of columns to load.

    Returns
    -------
    pl.DataFrame
        Daily-features panel keyed by (ticker, trade_date).
    """
    return (
        scan_daily_features(
            ticker,
            proc_root=proc_root,
            columns=columns,
        )
        .collect()
    )


def join_daily_features(
    lf: pl.LazyFrame,
    *,
    ticker: str,
    proc_root: Path | str = PROC_ORATS_DAILY_FEATURES,
    columns: Sequence[str] | None = None,
    how: str = "left",
) -> pl.LazyFrame:
    """Convenience helper: join daily features onto an existing LazyFrame.

    Assumes `lf` contains keys: (ticker, trade_date).

    Parameters
    ----------
    lf:
        Any LazyFrame containing (ticker, trade_date).
    ticker:
        Which daily-features panel to load.
    proc_root:
        Root directory of the processed daily-features dataset.
    columns:
        Optional subset of daily-features columns to join in.
        If None, joins all columns from the daily-features panel.
    how:
        Join type. Default "left" so your existing panel remains the base.

    Returns
    -------
    pl.LazyFrame
        `lf` with daily-features columns appended.
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