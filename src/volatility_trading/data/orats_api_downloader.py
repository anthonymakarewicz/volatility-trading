from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable

import polars as pl
import requests

from .orats_api_endpoints import ENDPOINTS, DownloadStrategy
from .orats_client_api import OratsClient

MAX_PER_CALL: int = 10


def chunk_tickers(tickers: Sequence[str]) -> list[list[str]]:
    """Split tickers into chunks of <= MAX_PER_CALL after stripping whitespace and dropping empties."""
    clean = [str(t).strip() for t in tickers if t is not None and str(t).strip()]
    return [clean[i : i + MAX_PER_CALL] for i in range(0, len(clean), MAX_PER_CALL)]


# ----------------------
# Helpers
# ----------------------

def _iter_weekday_trade_dates_for_year(year: int) -> list[str]:
    """Fallback trading calendar: all weekdays in a given year (Mon-Fri)."""

    """
    # Official US equity trading days (XNYS)
    cal = xcals.get_calendar("XNYS")
    schedule = cal.schedule.loc[start:end]

    trading_days = schedule.index.date.tolist()
    """
    start = dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    out: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += one
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write parquet atomically (best-effort) to avoid partial files on crashes."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def _raw_path_by_trade_date(raw_root: Path, endpoint: str, trade_date: str, part: int) -> Path:
    year = int(trade_date[:4])
    return (
        raw_root
        / f"endpoint={endpoint}"
        / f"year={year}"
        / f"tradeDate={trade_date}"
        / f"part={part:03d}.parquet"
    )


def _raw_path_full_history(raw_root: Path, endpoint: str, ticker: str) -> Path:
    return raw_root / f"endpoint={endpoint}" / f"underlying={ticker}" / "full.parquet"


def _supported_endpoints_str() -> str:
    return ", ".join(sorted(ENDPOINTS.keys()))


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
        params: dict[str, Any] = {"ticker": [ticker]}
        if fields_list is not None:
            params["fields"] = fields_list

        df = client.get_df(endpoint=endpoint, params=params, session=session)
        if df.height == 0:
            continue

        out_path = _raw_path_full_history(raw_root, endpoint, ticker)
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
    chunks = chunk_tickers(tickers)
    fields_list = list(fields) if fields else None

    params_base: dict[str, Any] = {}
    if fields_list is not None:
        params_base["fields"] = fields_list

    for year in years:
        trade_dates = _iter_weekday_trade_dates_for_year(year)
        for trade_date in trade_dates:
            for part, ticker_chunk in enumerate(chunks):
                params = dict(params_base)
                params["tradeDate"] = trade_date
                params["ticker"] = ticker_chunk  # ORATS expects 'ticker' (singular)

                df = client.get_df(endpoint=endpoint, params=params, session=session)
                if df.height == 0:
                    continue

                out_path = _raw_path_by_trade_date(raw_root, endpoint, trade_date, part)
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
      - BY_TRADE_DATE:   raw_root/endpoint=<endpoint>/year=YYYY/tradeDate=YYYY-MM-DD/part=XXX.parquet
      - FULL_HISTORY:    raw_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet
    """
    raw_root = Path(raw_root)

    tickers_clean = [str(t).strip() for t in tickers if t and str(t).strip()]
    if not tickers_clean:
        raise ValueError("tickers must be non-empty")

    try:
        spec = ENDPOINTS[endpoint]
    except KeyError as e:
        raise ValueError(f"Unknown endpoint '{endpoint}'. Supported: {_supported_endpoints_str()}") from e

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