"""I/O helpers for processed yfinance time-series datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_YFINANCE_TIME_SERIES


def yfinance_time_series_path(
    proc_root: Path | str = PROC_YFINANCE_TIME_SERIES,
) -> Path:
    """Build the processed yfinance time-series parquet path."""
    return Path(proc_root) / "yfinance_time_series.parquet"


def scan_yfinance_time_series(
    *,
    proc_root: Path | str = PROC_YFINANCE_TIME_SERIES,
    columns: Sequence[str] | None = None,
    tickers: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan processed yfinance time-series parquet lazily."""
    path = yfinance_time_series_path(proc_root)
    if not path.exists():
        raise FileNotFoundError(f"Processed yfinance time-series not found: {path}")

    lf = pl.scan_parquet(path)
    if tickers is not None:
        ticker_candidates: set[str] = set()
        for ticker in tickers:
            normalized = str(ticker).strip().upper()
            if not normalized:
                continue
            ticker_candidates.add(normalized)
            ticker_candidates.add(normalized.removeprefix("^"))
        ticker_list = sorted(t for t in ticker_candidates if t)
        if not ticker_list:
            raise ValueError("tickers must contain at least one non-empty symbol")
        lf = lf.filter(pl.col("ticker").is_in(ticker_list))
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_yfinance_time_series(
    *,
    proc_root: Path | str = PROC_YFINANCE_TIME_SERIES,
    columns: Sequence[str] | None = None,
    tickers: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read processed yfinance time-series parquet eagerly."""
    return scan_yfinance_time_series(
        proc_root=proc_root,
        columns=columns,
        tickers=tickers,
    ).collect()
