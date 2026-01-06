"""ORATS API downloader.

This module downloads ORATS API endpoints and snapshots the *raw JSON payloads*
to disk.

The downloader dispatches to a strategy based on the endpoint specification:
- BY_TRADE_DATE: one request per trading date and ticker chunk
- FULL_HISTORY: one request per ticker (no tradeDate)
"""
from __future__ import annotations

import datetime as dt
import io
import os
import gzip
import json
import logging
import time
import tempfile
from collections.abc import Callable, Sequence, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import requests

from .endpoints import DownloadStrategy, get_endpoint_spec
from .io import (
    ALLOWED_COMPRESSIONS,
    DEFAULT_COMPRESSION,
    ensure_dir,
    raw_path_by_trade_date,
    raw_path_full_history,
    validate_years,
)
from .client import OratsClient

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

MAX_PER_CALL: int = 10

# Progress logging cadence (kept small to avoid log spam)
LOG_EVERY_N_DATES: int = 25
LOG_EVERY_N_TICKERS: int = 10


# ----------------------------------------------------------------------------
# Downloader result dataclass
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadApiResult:
    """Summary of a download run."""
    endpoint: str
    strategy: DownloadStrategy
    n_requests_total: int
    n_written: int
    n_skipped: int
    n_empty_payloads: int
    n_failed: int
    duration_s: float
    out_paths: list[Path]
    failed_paths: list[Path]


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


def _get_trading_days(year: int) -> list[str]:
    """Get the NYSE trading sessions for a given year (XNYS calendar)."""
    cal = xcals.get_calendar("XNYS")
    start = dt.date(year, 1, 1).isoformat()
    end = dt.date(year, 12, 31).isoformat()
    sessions = cal.sessions_in_range(start, end)
    return [d.date().isoformat() for d in sessions]


def _http_status_from_error(err: BaseException) -> int | None:
    """Best-effort extract an HTTP status code from an exception."""
    if isinstance(err, requests.exceptions.HTTPError):
        resp = getattr(err, "response", None)
        if resp is not None:
            return getattr(resp, "status_code", None)
        return None

    # Our client raises RuntimeError(... ) from the last underlying error.
    if isinstance(err, RuntimeError):
        cause = err.__cause__
        if cause is not None and isinstance(cause, BaseException):
            return _http_status_from_error(cause)

    return None


def _is_fatal_download_error(err: BaseException) -> bool:
    """Return True for errors that should stop the run.

    We fail-fast on configuration/contract/auth errors (e.g., bad params/token,
    non-JSON responses) and only continue on transient server/network failures.
    """
    # Obvious programming/config issues.
    if isinstance(err, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return True

    # If the underlying cause is fatal, treat the wrapper as fatal.
    if isinstance(err, RuntimeError):
        cause = err.__cause__
        if cause is not None and isinstance(cause, BaseException):
            return _is_fatal_download_error(cause)

    status = _http_status_from_error(err)

    # Non-429 4xx are permanent client errors => stop.
    if status is not None and 400 <= status <= 499 and status != 429:
        return True

    # Otherwise treat as transient only if it's clearly a transport/server issue.
    if status is not None and (status == 429 or 500 <= status <= 599):
        return False

    if isinstance(
        err, (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return False

    # Unknown exception types: be conservative and stop.
    return True


def _write_json_atomic(
    payload: dict[str, Any],
    path: Path,
    *,
    compression: str,
) -> None:
    """
    Write a JSON payload atomically using a temp file + rename.

    This prevents partially-written *final* files by writing to a unique temp
    file in the same directory (same filesystem) and then swapping it into
    place with `os.replace`.

    We also `flush` + `os.fsync` the underlying file to improve durability
    against crashes/power loss (at some performance cost). Directory fsync is
    attempted best-effort on POSIX.
    """
    ensure_dir(path.parent)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    tmp_path: str | None = None

    try:
        # Unique temp file in the same directory => atomic swap works.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as f:
            tmp_path = f.name

        if tmp_path is None:
            raise RuntimeError("Failed to allocate a temporary file")

        if compression == "gz":
            # Write gzip-compressed UTF-8 JSON and fsync the underlying file.
            with open(tmp_path, mode="wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8") as txt:
                        txt.write(json.dumps(payload, ensure_ascii=False))
                        txt.write("\n")
                        txt.flush()
                raw.flush()
                os.fsync(raw.fileno())
        else:
            # Plain JSON
            with open(tmp_path, mode="w", encoding="utf-8") as txt:
                txt.write(json.dumps(payload, ensure_ascii=False))
                txt.write("\n")
                txt.flush()
                os.fsync(txt.fileno())

        os.replace(tmp_path, path)

        # Best-effort directory fsync (POSIX). Helps make the rename durable.
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except OSError:
            dir_fd = None

        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    except Exception:
        # Best-effort cleanup of temp file on failure.
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


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
    compression: str,
) -> DownloadApiResult:
    """Download a FULL_HISTORY endpoint (one request per ticker) and write JSON."""
    fields_list = list(fields) if fields else None

    t0 = time.perf_counter()
    out_paths: list[Path] = []
    failed_paths: list[Path] = []
    n_written = 0
    n_skipped = 0
    n_empty_payloads = 0
    n_failed = 0
    n_requests_total = 0

    logger.info(
        "Starting FULL_HISTORY download endpoint=%s tickers=%d fields=%s",
        endpoint,
        len(tickers),
        (len(fields_list) if fields_list is not None else "ALL"),
    )

    for ticker in tickers:
        out_path = raw_path_full_history(
            raw_root=raw_root,
            endpoint=endpoint,
            ticker=ticker,
            compression=compression,
        )

        if (not overwrite) and out_path.exists():
            # Skip existing files to avoid redundant downloads
            n_skipped += 1
            logger.debug(
                "Skipping existing file endpoint=%s ticker=%s path=%s",
                endpoint,
                ticker,
                out_path,
            )
            continue

        params: dict[str, Any] = {"ticker": [ticker]}
        if fields_list is not None:
            params["fields"] = fields_list

        n_requests_total += 1
        try:
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

            _write_json_atomic(payload, out_path, compression=compression)
            out_paths.append(out_path)
            n_written += 1

        except Exception as e:
            failed_paths.append(out_path)
            logger.exception(
                "Failed FULL_HISTORY endpoint=%s ticker=%s path=%s",
                endpoint,
                ticker,
                out_path,
            )

            if _is_fatal_download_error(e):
                raise

            n_failed += 1
            continue

        if (n_written % LOG_EVERY_N_TICKERS) == 0 and n_written > 0:
            logger.info(
                "Progress FULL_HISTORY endpoint=%s written=%d skipped=%d "
                "empty_payloads=%d failed=%d",
                endpoint,
                n_written,
                n_skipped,
                n_empty_payloads,
                n_failed,
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    duration_s = time.perf_counter() - t0
    result = DownloadApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.FULL_HISTORY,
        n_requests_total=n_requests_total,
        n_written=n_written,
        n_skipped=n_skipped,
        n_empty_payloads=n_empty_payloads,
        n_failed=n_failed,
        duration_s=duration_s,
        out_paths=out_paths,
        failed_paths=failed_paths,
    )

    logger.info(
        "Finished FULL_HISTORY endpoint=%s written=%d skipped=%d "
        "empty_payloads=%d failed=%d duration=%.2fs",
        endpoint,
        result.n_written,
        n_skipped,
        n_empty_payloads,
        n_failed,
        duration_s,
    )

    return result


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
    compression: str,
) -> DownloadApiResult:
    """Download a BY_TRADE_DATE endpoint (years -> sessions -> ticker chunks)."""
    chunks = _chunk_tickers(tickers)
    fields_list = list(fields) if fields else None

    params_base: dict[str, Any] = {}
    if fields_list is not None:
        params_base["fields"] = fields_list

    t0 = time.perf_counter()
    out_paths: list[Path] = []
    failed_paths: list[Path] = []
    n_written = 0
    n_skipped = 0
    n_empty_payloads = 0
    n_failed = 0
    n_requests_total = 0

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
                out_path = raw_path_by_trade_date(
                    raw_root=raw_root,
                    endpoint=endpoint,
                    trade_date=trade_date,
                    part=part,
                    compression=compression,
                )

                if (not overwrite) and out_path.exists():
                    # Skip existing files to avoid redundant downloads
                    n_skipped += 1
                    logger.debug(
                        "Skipping existing file endpoint=%s path=%s",
                        endpoint,
                        out_path,
                    )
                    continue

                params = dict(params_base)
                params["tradeDate"] = trade_date
                params["ticker"] = ticker_chunk

                n_requests_total += 1
                try:
                    payload = client.get_payload(
                        endpoint=endpoint,
                        params=params,
                        session=session,
                    )

                    data = payload.get("data", [])
                    if not data:
                        n_empty_payloads += 1
                        logger.debug(
                            "Empty payload data. endpoint=%s "
                            "tradeDate=%s part=%d",
                            endpoint,
                            trade_date,
                            part,
                        )

                    _write_json_atomic(payload, out_path, compression=compression)
                    n_written += 1
                    out_paths.append(out_path)

                except Exception as e:
                    failed_paths.append(out_path)
                    logger.exception(
                        "Failed BY_TRADE_DATE endpoint=%s tradeDate=%s " 
                        "part=%d path=%s",
                        endpoint,
                        trade_date,
                        part,
                        out_path,
                    )

                    if _is_fatal_download_error(e):
                        raise

                    n_failed += 1
                    continue

                if sleep_s > 0:
                    time.sleep(sleep_s)

            if (td_i % LOG_EVERY_N_DATES) == 0:
                logger.info(
                    "Progress BY_TRADE_DATE endpoint=%s year=%s date=%s "
                    "written=%d skipped=%d empty_payloads=%d failed=%d",
                    endpoint,
                    year,
                    trade_date,
                    n_written,
                    n_skipped,
                    n_empty_payloads,
                    n_failed,
                )

    duration_s = time.perf_counter() - t0
    result = DownloadApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.BY_TRADE_DATE,
        n_requests_total=n_requests_total,
        n_written=n_written,
        n_skipped=n_skipped,
        n_empty_payloads=n_empty_payloads,
        n_failed=n_failed,
        duration_s=duration_s,
        out_paths=out_paths,
        failed_paths=failed_paths,
    )

    logger.info(
        "Finished BY_TRADE_DATE endpoint=%s written=%d skipped=%d "
        "empty_payloads=%d failed=%d duration=%.2fs",
        endpoint,
        n_written,
        n_skipped,
        n_empty_payloads,
        n_failed,
        duration_s,
    )

    return result


DOWNLOAD_HANDLERS: dict[DownloadStrategy, Callable[..., DownloadApiResult]] = {
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
    tickers: Iterable[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    fields: Sequence[str] | None = None,
    compression: str = DEFAULT_COMPRESSION,
    sleep_s: float = 0.0,
    overwrite: bool = False,
) -> DownloadApiResult:
    """
    Download an ORATS API endpoint and store raw JSON payload snapshots.

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
        raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_chunk000.json.gz

    - FULL_HISTORY:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/data.json.gz

    Parameters
    ----------
    token:
        ORATS API token.
    endpoint:
        Logical endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
    raw_root:
        Root folder where raw snapshots will be written.
    tickers:
        Sequence of tickers. Values are stripped, de-duplicated (preserving first
        occurrence), and then (for BY_TRADE_DATE) chunked into <= MAX_PER_CALL.
    year_whitelist:
        Years to download for BY_TRADE_DATE endpoints. Required for BY_TRADE_DATE
        and ignored for FULL_HISTORY.
    fields:
        Optional list of ORATS fields to request. If None, ORATS returns its
        default field set for the endpoint.
    compression:
        Compression mode for raw JSON snapshots: "gz" (gzip) or "none".
    sleep_s:
        Optional polite delay (seconds) between requests.
    overwrite:
        If False (default), existing output files are skipped. If True, existing
        files are replaced.

    Notes
    -----
    - JSON payloads are written even when `payload['data']` is empty.
    - No `.empty`/`.error` markers are written.
    - Transient failures are logged and recorded in the returned result; the
      downloader continues.
    - Fatal failures (e.g., non-429 4xx, bad params, non-JSON responses) are
      re-raised to stop the run.
    """
    raw_root = Path(raw_root)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

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

    # Resolve endpoint -> (path, required params, download strategy).
    spec = get_endpoint_spec(endpoint)
    client = OratsClient(token=token)

    handler = DOWNLOAD_HANDLERS.get(spec.strategy, None)
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
            return handler(
                client=client,
                session=session,
                endpoint=endpoint,
                raw_root=raw_root,
                tickers=tickers_clean,
                fields=fields,
                sleep_s=sleep_s,
                overwrite=overwrite,
                compression=compression,
            )

        # BY_TRADE_DATE endpoints require an explicit year whitelist.
        if years_list is None:
            raise ValueError(
                "year_whitelist must be provided for BY_TRADE_DATE endpoints"
            )
        years = validate_years(years_list)

        return handler(
            client=client,
            session=session,
            endpoint=endpoint,
            raw_root=raw_root,
            tickers=tickers_clean,
            years=years,
            fields=fields,
            sleep_s=sleep_s,
            overwrite=overwrite,
            compression=compression,
        )