"""Download ORATS API endpoints and store raw parquet snapshots."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import polars as pl
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
# Private Helpers
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
    """Split *already-clean* tickers into chunks of <= MAX_PER_CALL.

    Assumes tickers were preprocessed by `download()`:
      - stripped
      - non-empty
      - unique (stable order)
    """
    t = list(tickers)
    return [t[i : i + MAX_PER_CALL] for i in range(0, len(t), MAX_PER_CALL)]


def _validate_years(
    year_whitelist: Iterable[int | str],
    *,
    min_year: int = MIN_YEAR,
    max_year: int | None = None,
) -> list[int]:
    if max_year is None:
        max_year = dt.date.today().year

    years = [int(y) for y in year_whitelist]
    if not years:
        raise ValueError("year_whitelist must be non-empty")

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
    p.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write parquet atomically (best-effort) to avoid partial files."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def _empty_marker_path(parquet_path: Path) -> Path:
    """Marker path recording an empty API response for this parquet output."""
    # YYYY-MM-DD_000.parquet -> YYYY-MM-DD_000.empty
    # full.parquet          -> full.empty
    return parquet_path.with_suffix(".empty")


def _write_empty_marker(parquet_path: Path, meta: dict[str, Any]) -> None:
    """Create an '.empty' marker atomically.

    We write a small JSON payload so that an empty marker is inspectable and
    debuggable (and can be expired/ignored later if desired).
    """
    marker = _empty_marker_path(parquet_path)
    _ensure_dir(marker.parent)

    payload = dict(meta)
    payload.setdefault("created_at", _utc_now_iso())

    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", 
        encoding="utf-8"
    )
    tmp.replace(marker)


def _marker_should_skip(marker_path: Path, *, ttl_days: int | None) -> bool:
    """Return True if an existing empty marker should cause a skip.

    If ttl_days is None, markers never expire.
    If ttl_days is set, markers older than ttl_days are ignored (download retried).
    """
    if ttl_days is None:
        return True

    try:
        txt = marker_path.read_text(encoding="utf-8").strip()
        if not txt:
            return True
        payload = json.loads(txt)
        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            return True

        # Parse ISO time with optional 'Z'
        s = created_at.replace("Z", "+00:00")
        created = dt.datetime.fromisoformat(s)
        now = dt.datetime.now(dt.UTC)
        age = now - created
        return age <= dt.timedelta(days=int(ttl_days))
    except Exception:
        # If marker is unreadable, be conservative and skip.
        return True


def _empty_marker_exists_and_valid(
    parquet_path: Path,
    *,
    ttl_days: int | None,
) -> bool:
    marker = _empty_marker_path(parquet_path)
    if not marker.exists():
        return False
    return _marker_should_skip(marker, ttl_days=ttl_days)


def _raw_path_by_trade_date(
    raw_root: Path,
    endpoint: str,
    trade_date: str,
    part: int,
) -> Path:
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
    return (
        raw_root 
        / f"endpoint={endpoint}" 
        / f"underlying={ticker}" 
        / "full.parquet"
    )


def _download_full_history(
    *,
    client: OratsClient,
    session: requests.Session,
    endpoint: str,
    raw_root: Path,
    tickers: Sequence[str],
    fields: Sequence[str] | None,
    sleep_s: float,
    empty_marker_ttl_days: int | None,
    overwrite: bool,
) -> None:
    fields_list = list(fields) if fields else None

    n_written = 0
    n_skipped_parquet = 0
    n_skipped_empty = 0
    n_empty_marked = 0

    logger.info(
        "Starting FULL_HISTORY download endpoint=%s tickers=%d fields=%s",
        endpoint,
        len(tickers),
        (len(fields_list) if fields_list is not None else "ALL"),
    )

    for ticker in tickers:
        out_path = _raw_path_full_history(raw_root, endpoint, ticker)
        if (not overwrite) and out_path.exists():
            n_skipped_parquet += 1
            continue
        if (not overwrite) and _empty_marker_exists_and_valid(
            out_path, ttl_days=empty_marker_ttl_days
        ):
            n_skipped_empty += 1
            continue

        params: dict[str, Any] = {"ticker": [ticker]}
        if fields_list is not None:
            params["fields"] = fields_list

        logger.debug("Fetching endpoint=%s ticker=%s", endpoint, ticker)
        df = client.get_df(endpoint=endpoint, params=params, session=session)

        if df.height == 0:
            if overwrite and out_path.exists():
                out_path.unlink()
            _write_empty_marker(
                out_path,
                {
                    "endpoint": endpoint,
                    "strategy": "FULL_HISTORY",
                    "ticker": ticker,
                    "fields": (fields_list if fields_list is not None else "ALL"),
                },
            )
            n_empty_marked += 1
            logger.debug(
                "Empty result (marked). endpoint=%s ticker=%s",
                endpoint,
                ticker,
            )
            continue

        _write_parquet_atomic(df, out_path)
        marker = _empty_marker_path(out_path)
        if marker.exists():
            marker.unlink()
        n_written += 1
        logger.debug(
            "Wrote parquet. endpoint=%s ticker=%s rows=%d path=%s",
            endpoint,
            ticker,
            df.height,
            out_path,
        )

        if (n_written % LOG_EVERY_N_TICKERS) == 0:
            logger.info(
                "Progress FULL_HISTORY endpoint=%s written=%d "
                "skipped_parquet=%d skipped_empty=%d empty_marked=%d",
                endpoint,
                n_written,
                n_skipped_parquet,
                n_skipped_empty,
                n_empty_marked,
            )

        if sleep_s > 0:
            time.sleep(sleep_s)

    logger.info(
        "Finished FULL_HISTORY endpoint=%s written=%d skipped_parquet=%d "
        "skipped_empty=%d empty_marked=%d",
        endpoint,
        n_written,
        n_skipped_parquet,
        n_skipped_empty,
        n_empty_marked,
    )


# ----------------------------------------------------------------------------
# Download Handlers (private)
# ----------------------------------------------------------------------------

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
    empty_marker_ttl_days: int | None,
    overwrite: bool,
) -> None:
    chunks = _chunk_tickers(tickers)
    fields_list = list(fields) if fields else None

    params_base: dict[str, Any] = {}
    if fields_list is not None:
        params_base["fields"] = fields_list

    n_written = 0
    n_skipped_parquet = 0
    n_skipped_empty = 0
    n_empty_marked = 0

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
                    n_skipped_parquet += 1
                    continue
                if (not overwrite) and _empty_marker_exists_and_valid(
                    out_path, ttl_days=empty_marker_ttl_days
                ):
                    n_skipped_empty += 1
                    continue

                params = dict(params_base)
                params["tradeDate"] = trade_date
                params["ticker"] = ticker_chunk  # ORATS expects 'ticker'

                logger.debug(
                    "Fetching endpoint=%s tradeDate=%s part=%d tickers=%s",
                    endpoint,
                    trade_date,
                    part,
                    ",".join(ticker_chunk),
                )

                df = client.get_df(
                    endpoint=endpoint, 
                    params=params,
                    session=session
                )

                if df.height == 0:
                    if overwrite and out_path.exists():
                        out_path.unlink()
                    _write_empty_marker(
                        out_path,
                        {
                            "endpoint": endpoint,
                            "strategy": "BY_TRADE_DATE",
                            "tradeDate": trade_date,
                            "part": part,
                            "tickers": list(ticker_chunk),
                            "fields": (fields_list if fields_list is not None else "ALL"),
                        },
                    )
                    n_empty_marked += 1
                    logger.debug(
                        "Empty result (marked). endpoint=%s tradeDate=%s part=%d",
                        endpoint,
                        trade_date,
                        part,
                    )
                    continue

                _write_parquet_atomic(df, out_path)
                marker = _empty_marker_path(out_path)
                if marker.exists():
                    marker.unlink()
                n_written += 1
                logger.debug(
                    "Wrote parquet. endpoint=%s tradeDate=%s part=%d rows=%d "
                    "path=%s",
                    endpoint,
                    trade_date,
                    part,
                    df.height,
                    out_path,
                )

                if sleep_s > 0:
                    time.sleep(sleep_s)

            # Progress log every N dates (per year)
            if (td_i % LOG_EVERY_N_DATES) == 0:
                logger.info(
                    "Progress BY_TRADE_DATE endpoint=%s year=%s date=%s "
                    "written=%d skipped_parquet=%d skipped_empty=%d "
                    "empty_marked=%d",
                    endpoint,
                    year,
                    trade_date,
                    n_written,
                    n_skipped_parquet,
                    n_skipped_empty,
                    n_empty_marked,
                )

    logger.info(
        "Finished BY_TRADE_DATE endpoint=%s written=%d skipped_parquet=%d "
        "skipped_empty=%d empty_marked=%d",
        endpoint,
        n_written,
        n_skipped_parquet,
        n_skipped_empty,
        n_empty_marked,
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
    empty_marker_ttl_days: int | None = None,
    overwrite: bool = False,
) -> None:
    """Download ORATS API data into raw parquet files.

    Dispatches download strategy based on endpoint spec:
      - BY_TRADE_DATE: loops years -> trading sessions -> ticker chunks
      - FULL_HISTORY:  one call per ticker (no tradeDate)

    Raw layout:
      - BY_TRADE_DATE:
        raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_XXX.parquet
      - FULL_HISTORY:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet

    Empty responses create ".empty" marker files with metadata.
    The empty_marker_ttl_days parameter can be used to ignore old markers and retry.
    """
    raw_root = Path(raw_root)

    tickers_clean = [
        str(t).strip() for t in tickers 
        if t is not None and str(t).strip()
    ]
    tickers_clean = _unique_preserve_order(tickers_clean)
    if not tickers_clean:
        raise ValueError("tickers must be non-empty")

    logger.info(
        "Download requested endpoint=%s tickers=%d years=%s fields=%s "
        "raw_root=%s sleep_s=%s empty_marker_ttl_days=%s overwrite=%s",
        endpoint,
        len(tickers_clean),
        (list(year_whitelist) if year_whitelist is not None else None),
        (len(fields) if fields is not None else "ALL"),
        raw_root,
        sleep_s,
        empty_marker_ttl_days,
        overwrite,
    )

    spec = get_endpoint_spec(endpoint)

    handler = DOWNLOAD_HANDLERS.get(spec.strategy)
    if handler is None:
        raise ValueError(
            f"No download handler registered for strategy: {spec.strategy}"
        )

    client = OratsClient(token=token)

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
                empty_marker_ttl_days=empty_marker_ttl_days,
                overwrite=overwrite,
            )
            return

        # BY_TRADE_DATE
        if year_whitelist is None:
            raise ValueError(
                "year_whitelist must be provided for BY_TRADE_DATE endpoints"
            )
        years = _validate_years(year_whitelist)

        handler(
            client=client,
            session=session,
            endpoint=endpoint,
            raw_root=raw_root,
            tickers=tickers_clean,
            years=years,
            fields=fields,
            sleep_s=sleep_s,
            empty_marker_ttl_days=empty_marker_ttl_days,
            overwrite=overwrite,
        )