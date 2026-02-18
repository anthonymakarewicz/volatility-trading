"""Internal helpers for ORATS API raw JSON downloads."""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import requests

from ..io import ALLOWED_COMPRESSIONS, ensure_dir

# ORATS list-style param size limit.
MAX_PER_CALL: int = 10

# Progress logging cadence (kept small to avoid log spam).
LOG_EVERY_N_DATES: int = 25
LOG_EVERY_N_TICKERS: int = 10

# TODO(Config): Move constants above constants into a shared config for download and extract


def unique_preserve_order(items: list[str]) -> list[str]:
    """Return unique items while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def chunk_tickers(tickers: list[str]) -> list[list[str]]:
    """Split tickers into <= MAX_PER_CALL chunks for ORATS list-style params."""
    return [tickers[i : i + MAX_PER_CALL] for i in range(0, len(tickers), MAX_PER_CALL)]


def get_trading_days(year: int) -> list[str]:
    """Get the NYSE trading sessions for a given year (XNYS calendar)."""
    cal = xcals.get_calendar("XNYS")
    start = dt.date(year, 1, 1).isoformat()
    end = dt.date(year, 12, 31).isoformat()
    sessions = cal.sessions_in_range(start, end)
    return [d.date().isoformat() for d in sessions]


def http_status_from_error(err: BaseException) -> int | None:
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
            return http_status_from_error(cause)

    return None


def is_fatal_download_error(err: BaseException) -> bool:
    """Return True for errors that should stop the run."""
    if isinstance(err, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return True

    if isinstance(err, RuntimeError):
        cause = err.__cause__
        if cause is not None and isinstance(cause, BaseException):
            return is_fatal_download_error(cause)

    status = http_status_from_error(err)

    # Non-429 4xx are permanent client errors => stop.
    if status is not None and 400 <= status <= 499 and status != 429:
        return True

    # 429 / 5xx are transient.
    if status is not None and (status == 429 or 500 <= status <= 599):
        return False

    if isinstance(
        err,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    ):
        return False

    # Unknown exception types: be conservative and stop.
    return True


def write_json_atomic(
    payload: dict[str, Any],
    path: Path,
    *,
    compression: str,
) -> None:
    """Write JSON atomically (temp file + rename), optionally gzip-compressed."""
    ensure_dir(path.parent)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    tmp_path: str | None = None

    try:
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
            with open(tmp_path, mode="wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8") as txt:
                        txt.write(json.dumps(payload, ensure_ascii=False))
                        txt.write("\n")
                        txt.flush()
                raw.flush()
                os.fsync(raw.fileno())
        else:
            with open(tmp_path, mode="w", encoding="utf-8") as txt:
                txt.write(json.dumps(payload, ensure_ascii=False))
                txt.write("\n")
                txt.flush()
                os.fsync(txt.fileno())

        os.replace(tmp_path, path)

        # Best-effort directory fsync (POSIX).
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
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
