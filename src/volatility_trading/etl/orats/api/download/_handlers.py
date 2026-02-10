"""Strategy-specific handlers for ORATS API download orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import requests

from ..client import OratsClient
from ..endpoints import DownloadStrategy
from ..io import raw_path_by_trade_date, raw_path_full_history
from ..types import DownloadApiResult
from ._helpers import (
    LOG_EVERY_N_DATES,
    LOG_EVERY_N_TICKERS,
    chunk_tickers,
    get_trading_days,
    is_fatal_download_error,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def download_full_history(
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
    """Download a FULL_HISTORY endpoint (one request per ticker)."""
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

            write_json_atomic(payload, out_path, compression=compression)
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

            if is_fatal_download_error(e):
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


def download_by_trade_date(
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
    chunks = chunk_tickers(list(tickers))
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
        trade_dates = get_trading_days(year)
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
                            "Empty payload data. endpoint=%s tradeDate=%s part=%d",
                            endpoint,
                            trade_date,
                            part,
                        )

                    write_json_atomic(payload, out_path, compression=compression)
                    n_written += 1
                    out_paths.append(out_path)

                except Exception as e:
                    failed_paths.append(out_path)
                    logger.exception(
                        "Failed BY_TRADE_DATE endpoint=%s tradeDate=%s part=%d path=%s",
                        endpoint,
                        trade_date,
                        part,
                        out_path,
                    )

                    if is_fatal_download_error(e):
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
    DownloadStrategy.FULL_HISTORY: download_full_history,
    DownloadStrategy.BY_TRADE_DATE: download_by_trade_date,
}
