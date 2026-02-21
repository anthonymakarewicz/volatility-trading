"""Sync yfinance OHLCV time-series data to raw and processed parquet datasets."""

from __future__ import annotations

from pathlib import Path

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
    output = frame.rename(columns=PRICE_COLUMNS)
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

    combined_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        frame = _download_ticker(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if frame.empty:
            continue

        raw_path = raw_root / f"{ticker.upper()}.parquet"
        if overwrite or not raw_path.exists():
            frame.to_parquet(raw_path, index=True)

        with_ticker = frame.copy()
        with_ticker["ticker"] = ticker.upper()
        with_ticker = with_ticker.reset_index()
        first_col = str(with_ticker.columns[0])
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
