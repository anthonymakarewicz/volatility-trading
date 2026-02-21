"""Sync yfinance OHLCV time-series data to raw and processed parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

INTERNAL_DOWNLOAD_TICKER_MAP = {
    "SP500TR": "^SP500TR",
    "VIX": "^VIX",
}


def _xs_columns_as_frame(frame: pd.DataFrame, *, key: str, level: int) -> pd.DataFrame:
    """Select one column level and always return a DataFrame."""
    selected = frame.xs(key, axis=1, level=level, drop_level=True)
    if isinstance(selected, pd.Series):
        return selected.to_frame()
    return selected


def _normalize_download_columns(frame: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to single-level columns for one symbol."""
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    out = frame.copy()
    cols = out.columns

    if cols.nlevels == 2:
        level0 = cols.get_level_values(0)
        level1 = cols.get_level_values(1)

        if level1.nunique() == 1:
            out.columns = level0
            return out
        if level0.nunique() == 1:
            out.columns = level1
            return out
        if ticker in set(level1):
            return _xs_columns_as_frame(out, key=ticker, level=1)
        if ticker in set(level0):
            return _xs_columns_as_frame(out, key=ticker, level=0)

    multi_columns = cast(pd.MultiIndex, out.columns)
    out.columns = [
        "_".join(str(part) for part in col if str(part))
        for col in multi_columns.to_flat_index()
    ]
    return out


def _normalize_input_ticker(ticker: str) -> str:
    normalized = str(ticker).strip().upper()
    if not normalized:
        raise ValueError("tickers must contain non-empty symbols")
    return normalized


def _resolve_download_ticker(user_ticker: str) -> str:
    if user_ticker.startswith("^"):
        return user_ticker
    return INTERNAL_DOWNLOAD_TICKER_MAP.get(user_ticker, user_ticker)


def _storage_ticker_from_download_ticker(download_ticker: str) -> str:
    storage_ticker = download_ticker.removeprefix("^")
    if not storage_ticker:
        raise ValueError("ticker resolved to empty storage symbol")
    return storage_ticker


def _build_ticker_plan(tickers: list[str]) -> list[tuple[str, str]]:
    """Return ordered unique `(download_ticker, storage_ticker)` pairs."""
    storage_to_download: dict[str, str] = {}
    for ticker in tickers:
        user_ticker = _normalize_input_ticker(ticker)
        download_ticker = _resolve_download_ticker(user_ticker)
        storage_ticker = _storage_ticker_from_download_ticker(download_ticker)

        existing_download = storage_to_download.get(storage_ticker)
        if existing_download is None:
            storage_to_download[storage_ticker] = download_ticker
            continue
        if existing_download != download_ticker:
            raise ValueError(
                "Ambiguous ticker mapping for storage symbol "
                f"'{storage_ticker}': '{existing_download}' vs '{download_ticker}'."
            )

    return [(download, storage) for storage, download in storage_to_download.items()]


def _download_ticker(
    *,
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str,
    auto_adjust: bool,
    actions: bool,
) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        progress=False,
    )
    if frame is None:
        return pd.DataFrame()
    if frame.empty:
        return frame
    output = _normalize_download_columns(frame, ticker=ticker)
    output.columns = [str(col) for col in output.columns]
    output = output.rename(columns=PRICE_COLUMNS)
    output.index = pd.to_datetime(output.index).tz_localize(None)
    return output.sort_index()


def sync_yfinance_time_series(
    *,
    tickers: list[str],
    raw_root: Path,
    proc_root: Path,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    actions: bool = False,
    overwrite: bool = False,
) -> Path:
    """Sync yfinance OHLCV data and return processed parquet path."""
    raw_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)
    ticker_plan = _build_ticker_plan(tickers)

    combined_frames: list[pd.DataFrame] = []
    for download_ticker, storage_ticker in ticker_plan:
        frame = _download_ticker(
            ticker=download_ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if frame.empty:
            continue

        raw_path = raw_root / f"{storage_ticker}.parquet"
        if overwrite or not raw_path.exists():
            frame.to_parquet(raw_path, index=True)

        with_ticker = frame.copy()
        with_ticker["ticker"] = storage_ticker
        with_ticker = with_ticker.reset_index()
        first_col = with_ticker.columns[0]
        with_ticker = with_ticker.rename(columns={first_col: "date"})
        combined_frames.append(with_ticker)

    if combined_frames:
        processed = pd.concat(combined_frames, ignore_index=True).sort_values(
            ["date", "ticker"]
        )
    else:
        processed = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    proc_path = proc_root / "yfinance_time_series.parquet"
    if overwrite or not proc_path.exists():
        processed.to_parquet(proc_path, index=False)
    return proc_path
