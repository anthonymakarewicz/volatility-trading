"""ORATS API downloader.

This module downloads ORATS API endpoints and snapshots the *raw JSON payloads*
to disk. It is intended as an ingestion layer:
- raw = "exactly what ORATS returned" (one JSON file per request)
- downstream code can build curated parquet datasets in intermediate/processed

The downloader dispatches to a strategy based on the endpoint specification:
- BY_TRADE_DATE: one request per trading date and ticker chunk
- FULL_HISTORY: one request per ticker (no tradeDate)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import requests

from .orats_api_endpoints import DownloadStrategy, get_endpoint_spec
from .orats_client_api import OratsClient

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

MAX_PER_CALL: int = 10
MIN_YEAR: int = 2007

# Progress logging cadence (kept small to avoid log spam)
LOG_EVERY_N_DATES: int = 25
LOG_EVERY_N_TICKERS: int = 10


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _unique_preserve_order(items: Sequence[str]) -> list[str]:
    """Return unique items while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _chunk_tickers(tickers: Sequence[str]) -> list[list[str]]:
    """Split tickers into <= MAX_PER_CALL chunks for ORATS list-style params."""
    t = list(tickers)
    return [t[i : i + MAX_PER_CALL] for i in range(0, len(t), MAX_PER_CALL)]


def _validate_years(
    years: Iterable[int | str],
    *,
    min_year: int = MIN_YEAR,
    max_year: int | None = None,
) -> list[int]:
    """Validate and coerce the year whitelist into a bounded list of ints."""
    if max_year is None:
        max_year = dt.date.today().year
        
    bad = [y for y in years if y < min_year or y > max_year]
    if bad:
        raise ValueError(
            f"Invalid years {bad}. Expected range [{min_year}, {max_year}]."
        )

    return years


def _get_trading_days(year: int) -> list[str]:
    """Get the NYSE trading sessions for a given year (XNYS calendar)."""
    cal = xcals.get_calendar("XNYS")
    start = dt.date(year, 1, 1).isoformat()
    end = dt.date(year, 12, 31).isoformat()
    sessions = cal.sessions_in_range(start, end)
    return [d.date().isoformat() for d in sessions]


def _ensure_dir(p: Path) -> None:
    """Create a directory (and parents) if needed."""
    p.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON payload atomically (best-effort) to avoid partial files."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _raw_path_by_trade_date(
    raw_root: Path,
    endpoint: str,
    trade_date: str,
    part: int,
) -> Path:
    """Raw file path for trade-date based endpoints.

    Layout:
      raw_root/endpoint=<endpoint>/year=YYYY/<tradeDate>_<part>.json
    """
    year = int(trade_date[:4])
    return (
        raw_root
        / f"endpoint={endpoint}"
        / f"year={year}"
        / f"{trade_date}_{part:03d}.json"
    )


def _raw_path_full_history(raw_root: Path, endpoint: str, ticker: str) -> Path:
    """Raw file path for full-history endpoints (one file per ticker)."""
    return (
        raw_root
        / f"endpoint={endpoint}"
        / f"underlying={ticker}"
        / "full.json"
    )


# ----------------------------------------------------------------------------
# Download handlers (private)
# ----------------------------------------------------------------------------

def _download_full_history(
    *,
    client: OratsClient,
    session: requests.Session,
    endpoint: str,
    raw_root: Path,
    tickers: Sequence[str],
    fields: Sequence[str] | None,
    sleep_s: float,
    overwrite: bool,
) -> None:
    """Download a FULL_HISTORY endpoint (one request per ticker) and write JSON."""
    fields_list = list(fields) if fields else None

    n_written = 0
    n_skipped = 0
    n_empty_payloads = 0

    logger.info(
        "Starting FULL_HISTORY download endpoint=%s tickers=%d fields=%s",
        endpoint,
        len(tickers),
        (len(fields_list) if fields_list is not None else "ALL"),
    )

    for ticker in tickers:
        out_path = _raw_path_full_history(raw_root, endpoint, ticker)
        if (not overwrite) and out_path.exists():
            # Skip existing files to avoid redundant downloads
            n_skipped += 1
            continue

        params: dict[str, Any] = {"ticker": [ticker]}
        if fields_list is not None:
            params["fields"] = fields_list

        # Propagate any HTTP exception
        payload = client.get_payload(
            endpoint=endpoint,
            params=params,
            session=session,
        )

        data = payload.get("data", [])
        if not data:
            n_empty_payloads += 1
            logger.debug(
                "Empty payload data. endpoint=%s ticker=%s",
                endpoint,
                ticker,
            )

        _write_json_atomic(payload, out_path)
        n_written += 1

        if (n_written % LOG_EVERY_N_TICKERS) == 0:
            logger.info(
                "Progress FULL_HISTORY endpoint=%s written=%d skipped=%d "
                "empty_payloads=%d",
                endpoint,
                n_written,
                n_skipped,
                n_empty_payloads,
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    logger.info(
        "Finished FULL_HISTORY endpoint=%s written=%d skipped=%d "
        "empty_payloads=%d",
        endpoint,
        n_written,
        n_skipped,
        n_empty_payloads,
    )


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
    overwrite: bool,
) -> None:
    """Download a BY_TRADE_DATE endpoint (years -> sessions -> ticker chunks)."""
    chunks = _chunk_tickers(tickers)
    fields_list = list(fields) if fields else None

    params_base: dict[str, Any] = {}
    if fields_list is not None:
        params_base["fields"] = fields_list

    n_written = 0
    n_skipped = 0
    n_empty_payloads = 0

    logger.info(
        "Starting BY_TRADE_DATE download endpoint=%s years=%d tickers=%d "
        "chunks=%d fields=%s",
        endpoint,
        len(years),
        len(tickers),
        len(chunks),
        (len(fields_list) if fields_list is not None else "ALL"),
    )

    for year in years:
        trade_dates = _get_trading_days(year)
        logger.info(
            "Year %s: %d trading days (XNYS) endpoint=%s",
            year,
            len(trade_dates),
            endpoint,
        )

        for td_i, trade_date in enumerate(trade_dates, start=1):
            for part, ticker_chunk in enumerate(chunks):
                out_path = _raw_path_by_trade_date(
                    raw_root=raw_root,
                    endpoint=endpoint,
                    trade_date=trade_date,
                    part=part,
                )

                if (not overwrite) and out_path.exists():
                    # Skip existing files to avoid redundant downloads
                    n_skipped += 1
                    continue

                params = dict(params_base)
                params["tradeDate"] = trade_date
                params["ticker"] = ticker_chunk

                # Propagate any HTTP exception
                payload = client.get_payload(
                    endpoint=endpoint,
                    params=params,
                    session=session,
                )

                data = payload.get("data", [])
                if not data:
                    n_empty_payloads += 1
                    logger.debug(
                        "Empty payload data. endpoint=%s tradeDate=%s part=%d",
                        endpoint,
                        trade_date,
                        part,
                    )

                _write_json_atomic(payload, out_path)
                n_written += 1

                if sleep_s > 0:
                    time.sleep(sleep_s)

            if (td_i % LOG_EVERY_N_DATES) == 0:
                logger.info(
                    "Progress BY_TRADE_DATE endpoint=%s year=%s date=%s "
                    "written=%d skipped=%d empty_payloads=%d",
                    endpoint,
                    year,
                    trade_date,
                    n_written,
                    n_skipped,
                    n_empty_payloads,
                )

    logger.info(
        "Finished BY_TRADE_DATE endpoint=%s written=%d skipped=%d "
        "empty_payloads=%d",
        endpoint,
        n_written,
        n_skipped,
        n_empty_payloads,
    )


DOWNLOAD_HANDLERS: dict[DownloadStrategy, Callable[..., None]] = {
    DownloadStrategy.FULL_HISTORY: _download_full_history,
    DownloadStrategy.BY_TRADE_DATE: _download_by_trade_date,
}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def download(
    *,
    token: str,
    endpoint: str,
    raw_root: str | Path,
    tickers: Sequence[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    fields: Sequence[str] | None = None,
    sleep_s: float = 0.0,
    overwrite: bool = False,
) -> None:
    """Download an ORATS API endpoint and store raw JSON payload snapshots.

    This is an ingestion utility that writes *one JSON file per API request*,
    preserving the response as returned by ORATS ("raw" layer). Downstream
    pipelines can later parse these snapshots into curated parquet datasets.

    Strategy
    --------
    The endpoint determines the download strategy via `get_endpoint_spec(endpoint)`:

    - BY_TRADE_DATE:
        Iterates over NYSE trading sessions (XNYS) for the requested years, and
        downloads one payload per (tradeDate, ticker_chunk). Tickers are chunked
        to respect ORATS list-parameter size limits.

    - FULL_HISTORY:
        Downloads one payload per ticker (no `tradeDate`). Any `year_whitelist`
        passed by the caller is ignored (and a warning is logged).

    Storage layout
    --------------
    - BY_TRADE_DATE:
        raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_XXX.json

    - FULL_HISTORY:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/full.json

    Parameters
    ----------
    token:
        ORATS API token.
    endpoint:
        Logical endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
    raw_root:
        Root folder where raw snapshots will be written.
    tickers:
        Iterable of tickers. Values are stripped, de-duplicated (preserving first
        occurrence), and then (for BY_TRADE_DATE) chunked into <= MAX_PER_CALL.
    year_whitelist:
        Years to download for BY_TRADE_DATE endpoints. Required for BY_TRADE_DATE
        and ignored for FULL_HISTORY.
    fields:
        Optional list of ORATS fields to request. If None, ORATS returns its
        default field set for the endpoint.
    sleep_s:
        Optional polite delay (seconds) between requests.
    overwrite:
        If False (default), existing output files are skipped. If True, existing
        files are replaced.

    Notes
    -----
    - JSON payloads are written even when `payload['data']` is empty.
    - No `.empty`/`.error` markers are written. If a request fails (non-OK HTTP or
      retries exhausted), an exception is raised and the run stops; the
      corresponding output JSON file will be missing.
    """
    raw_root = Path(raw_root)

    # Normalize tickers early so chunking and logging are stable.
    tickers_clean = [
        str(t).strip() for t in tickers if t is not None and str(t).strip()
    ]
    tickers_clean = _unique_preserve_order(tickers_clean)
    if not tickers_clean:
        raise ValueError("tickers must be non-empty")

    years_list: list[int] | None
    if year_whitelist is None:
        years_list = None
    else:
        years_list = [int(y) for y in year_whitelist]

    logger.info(
        "Download requested endpoint=%s tickers=%d years=%s fields=%s "
        "raw_root=%s sleep_s=%s overwrite=%s",
        endpoint,
        len(tickers_clean),
        years_list,
        (len(fields) if fields is not None else "ALL"),
        raw_root,
        sleep_s,
        overwrite,
    )

    # Resolve endpoint -> (path, required params, download strategy).
    spec = get_endpoint_spec(endpoint)
    client = OratsClient(token=token)

    handler = DOWNLOAD_HANDLERS.get(spec.strategy)
    if handler is None:
        raise ValueError(
            f"No download handler registered for strategy: {spec.strategy}"
        )

    # Shared session = connection reuse across many requests.
    with requests.Session() as session:
        if spec.strategy == DownloadStrategy.FULL_HISTORY:
            if year_whitelist is not None:
                logger.warning(
                    "year_whitelist is ignored for endpoint=%s (FULL_HISTORY)",
                    endpoint,
                )
            handler(
                client=client,
                session=session,
                endpoint=endpoint,
                raw_root=raw_root,
                tickers=tickers_clean,
                fields=fields,
                sleep_s=sleep_s,
                overwrite=overwrite,
            )
            return

        # BY_TRADE_DATE endpoints require an explicit year whitelist.
        if years_list is None:
            raise ValueError(
                "year_whitelist must be provided for BY_TRADE_DATE endpoints"
            )
        years = _validate_years(years_list)

        handler(
            client=client,
            session=session,
            endpoint=endpoint,
            raw_root=raw_root,
            tickers=tickers_clean,
            years=years,
            fields=fields,
            sleep_s=sleep_s,
            overwrite=overwrite,
        )