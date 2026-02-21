from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from volatility_trading.datasets.yfinance import (
    read_yfinance_time_series,
    scan_yfinance_time_series,
    yfinance_time_series_path,
)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_yfinance_time_series_path_builds_expected_location(tmp_path: Path) -> None:
    path = yfinance_time_series_path(tmp_path)
    assert path == tmp_path / "yfinance_time_series.parquet"


def test_scan_yfinance_time_series_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="Processed yfinance time-series not found"
    ):
        scan_yfinance_time_series(proc_root=tmp_path).collect()


def test_scan_yfinance_time_series_filters_tickers_and_columns(tmp_path: Path) -> None:
    path = yfinance_time_series_path(tmp_path)
    _write_parquet(
        path,
        pl.DataFrame(
            {
                "date": [dt.date(2022, 1, 3), dt.date(2022, 1, 3), dt.date(2022, 1, 4)],
                "ticker": ["SPY", "QQQ", "SPY"],
                "close": [475.0, 398.0, 476.5],
                "volume": [1_000, 2_000, 1_200],
            }
        ),
    )

    out = (
        scan_yfinance_time_series(
            proc_root=tmp_path,
            tickers=["spy"],
            columns=["date", "ticker", "close"],
        )
        .collect()
        .sort("date")
    )

    assert out.columns == ["date", "ticker", "close"]
    assert out.height == 2
    assert set(out["ticker"].to_list()) == {"SPY"}


def test_scan_yfinance_time_series_accepts_caret_prefixed_references(
    tmp_path: Path,
) -> None:
    path = yfinance_time_series_path(tmp_path)
    _write_parquet(
        path,
        pl.DataFrame(
            {
                "date": [dt.date(2022, 1, 3), dt.date(2022, 1, 4)],
                "ticker": ["SP500TR", "SP500TR"],
                "close": [5000.0, 5010.0],
            }
        ),
    )

    out = scan_yfinance_time_series(proc_root=tmp_path, tickers=["^SP500TR"]).collect()
    assert out.height == 2


def test_scan_yfinance_time_series_rejects_empty_tickers(tmp_path: Path) -> None:
    path = yfinance_time_series_path(tmp_path)
    _write_parquet(
        path,
        pl.DataFrame(
            {"date": [dt.date(2022, 1, 3)], "ticker": ["SPY"], "close": [1.0]}
        ),
    )

    with pytest.raises(
        ValueError, match="tickers must contain at least one non-empty symbol"
    ):
        scan_yfinance_time_series(proc_root=tmp_path, tickers=[" ", ""]).collect()


def test_read_yfinance_time_series_reads_full_frame(tmp_path: Path) -> None:
    path = yfinance_time_series_path(tmp_path)
    _write_parquet(
        path,
        pl.DataFrame(
            {
                "date": [dt.date(2023, 5, 1), dt.date(2023, 5, 2)],
                "ticker": ["SPY", "SPY"],
                "close": [410.0, 412.0],
            }
        ),
    )

    out = read_yfinance_time_series(proc_root=tmp_path)
    assert out.height == 2
