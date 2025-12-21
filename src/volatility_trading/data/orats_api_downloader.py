from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable

import exchange_calendars as xcals
import pandas as pd
import polars as pl
import requests

from .orats_api_endpoints import DownloadStrategy, get_endpoint_spec
from .orats_client_api import OratsClient

MAX_PER_CALL: int = 10


# ----------------------
# Helpers
# ----------------------

def _chunk_tickers(tickers: Sequence[str]) -> list[list[str]]:
    """Split tickers into chunks of <= MAX_PER_CALL after stripping whitespace and dropping empties."""
    clean = [str(t).strip() for t in tickers if t is not None and str(t).strip()]
    return [clean[i : i + MAX_PER_CALL] for i in range(0, len(clean), MAX_PER_CALL)]


def _get_trading_days(year: int) -> list[str]:
    """Get the trading days for the NYSE for a given year"""
    cal = xcals.get_calendar("XNYS")
    start = pd.Timestamp(f"{year}-01-01")
    end   = pd.Timestamp(f"{year}-12-31")
    sessions = cal.sessions_in_range(start, end)  # DatetimeIndex
    return [d.date().isoformat() for d in sessions]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write parquet atomically (best-effort) to avoid partial files on crashes."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def _empty_marker_path(parquet_path: Path) -> Path:
    """Return the marker path used to record an empty API response for this parquet output."""
    # YYYY-MM-DD_000.parquet -> YYYY-MM-DD_000.empty
    return parquet_path.with_suffix(".empty")


def _write_empty_marker(parquet_path: Path) -> None:
    """Create an '.empty' marker atomically to avoid re-hitting the API on reruns."""
    marker = _empty_marker_path(parquet_path)
    _ensure_dir(marker.parent)
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text("empty\n", encoding="utf-8")
    tmp.replace(marker)


def _raw_path_by_trade_date(raw_root: Path, endpoint: str, trade_date: str, part: int) -> Path:
    """Raw file path for trade-date based endpoints.
    Layout:
      raw_root/endpoint=<endpoint>/year=YYYY/<tradeDate>_<part>.parquet
    """
    year = int(trade_date[:4])
    return (
        raw_root
        / f"endpoint={endpoint}"
        / f"year={year}"
        / f"{trade_date}_{part:03d}.parquet"
    )


def _raw_path_full_history(raw_root: Path, endpoint: str, ticker: str) -> Path:
    return raw_root / f"endpoint={endpoint}" / f"underlying={ticker}" / "full.parquet"


# ----------------------
# Download handlers (private)
# ----------------------

def _download_full_history(
    *,
    client: OratsClient,
    session: requests.Session,
    endpoint: str,
    raw_root: Path,
    tickers: Sequence[str],
    fields: Sequence[str] | None,
    sleep_s: float,
) -> None:
    fields_list = list(fields) if fields else None

    for ticker in tickers:
        # Skip if already downloaded as parquet OR known-empty
        out_path = _raw_path_full_history(raw_root, endpoint, ticker)
        if out_path.exists() or _empty_marker_path(out_path).exists():
            continue

        params: dict[str, Any] = {"ticker": [ticker]}
        if fields_list is not None:
            params["fields"] = fields_list

        df = client.get_df(endpoint=endpoint, params=params, session=session)
        if df.height == 0:
            _write_empty_marker(out_path)
            continue
        _write_parquet_atomic(df, out_path)

        if sleep_s > 0:
            time.sleep(sleep_s)


def _download_by_trade_date(
    *,
    client: OratsClient,
    session: requests.Session,
    endpoint: str,
    raw_root: Path,
    tickers: Sequence[str],
    years: Sequence[int],
    fields: Sequence[str] | None,
    sleep_s: float,
) -> None:
    chunks = _chunk_tickers(tickers)
    fields_list = list(fields) if fields else None

    params_base: dict[str, Any] = {}
    if fields_list is not None:
        params_base["fields"] = fields_list

    for year in years:
        trade_dates = _get_trading_days(year)
        for trade_date in trade_dates:
            for part, ticker_chunk in enumerate(chunks):

                # Skip if already downloaded as parquet OR known-empty
                out_path = _raw_path_by_trade_date(raw_root, endpoint, trade_date, part)
                if out_path.exists() or _empty_marker_path(out_path).exists():
                    continue

                params = dict(params_base)
                params["tradeDate"] = trade_date
                params["ticker"] = ticker_chunk  # ORATS expects 'ticker' (singular)

                df = client.get_df(endpoint=endpoint, params=params, session=session)
                if df.height == 0:
                    _write_empty_marker(out_path)
                    continue
                _write_parquet_atomic(df, out_path)

                if sleep_s > 0:
                    time.sleep(sleep_s)


DOWNLOAD_HANDLERS: dict[DownloadStrategy, Callable[..., None]] = {
    DownloadStrategy.FULL_HISTORY: _download_full_history,
    DownloadStrategy.BY_TRADE_DATE: _download_by_trade_date,
}


# ----------------------
# Public API
# ----------------------

def download(
    *,
    token: str,
    endpoint: str,
    raw_root: str | Path,
    tickers: Sequence[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    fields: Sequence[str] | None = None,
    sleep_s: float = 0.0,
) -> None:
    """Download ORATS API data into raw parquet files.

    Dispatches download strategy based on endpoint spec:
      - DownloadStrategy.BY_TRADE_DATE: loops years -> weekdays -> ticker chunks
      - DownloadStrategy.FULL_HISTORY: one call per ticker (no tradeDate)

    Raw layout:
      - BY_TRADE_DATE:   raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_XXX.parquet
      - FULL_HISTORY:    raw_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet
    """
    raw_root = Path(raw_root)

    tickers_clean = [str(t).strip() for t in tickers if t and str(t).strip()]
    if not tickers_clean:
        raise ValueError("tickers must be non-empty")

    spec = get_endpoint_spec(endpoint)

    handler = DOWNLOAD_HANDLERS.get(spec.strategy)
    if handler is None:
        raise ValueError(f"No download handler registered for strategy: {spec.strategy}")

    client = OratsClient(token=token)

    with requests.Session() as session:
        if spec.strategy == DownloadStrategy.FULL_HISTORY:
            if year_whitelist is not None:
                print(f"year_whitelist is ignored for {endpoint}. All history is downloaded")
            handler(
                client=client,
                session=session,
                endpoint=endpoint,
                raw_root=raw_root,
                tickers=tickers_clean,
                fields=fields,
                sleep_s=sleep_s,
            )
            return

        # BY_TRADE_DATE
        if year_whitelist is None:
            raise ValueError("year_whitelist must be provided for BY_TRADE_DATE endpoints")
        years = [int(y) for y in year_whitelist]
        if not years:
            raise ValueError("year_whitelist must be non-empty")

        handler(
            client=client,
            session=session,
            endpoint=endpoint,
            raw_root=raw_root,
            tickers=tickers_clean,
            years=years,
            fields=fields,
            sleep_s=sleep_s,
        )