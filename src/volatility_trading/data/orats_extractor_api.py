# src/volatility_trading/data/orats_extractor_api.py
"""ORATS API extractor (raw JSON snapshots -> intermediate parquet).

This module converts ORATS API raw snapshots (stored as JSON/JSON.GZ) into
intermediate parquet files, partitioned for fast downstream usage.

The raw layout is assumed to match `orats_downloader_api.py`:

FULL_HISTORY endpoints:
    raw_root/endpoint=<endpoint>/underlying=<TICKER>/data.json(.gz)

BY_TRADE_DATE endpoints:
    raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_chunkXXX.json(.gz)

Intermediate output layout (default):
    intermediate_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet

Notes:
    - FULL_HISTORY raw is already ticker-centric.
    - BY_TRADE_DATE raw is date-centric, but intermediate is ticker-centric for
      downstream joins/usage.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .orats_api_endpoints import DownloadStrategy, get_endpoint_spec

logger = logging.getLogger(__name__)

MIN_YEAR: int = 2007
ALLOWED_COMPRESSIONS: set[str] = {"gz", "none"}


# ----------------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractApiResult:
    """Summary of an extraction run."""
    endpoint: str
    strategy: DownloadStrategy
    n_raw_files_seen: int
    n_raw_files_read: int
    n_failed: int
    n_rows_total: int
    out_paths: list[Path]
    failed_paths: list[Path]


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    """Create a directory (and parents) if needed."""
    p.mkdir(parents=True, exist_ok=True)


def _validate_years(
    years: list[int],
    *,
    min_year: int = MIN_YEAR,
    max_year: int | None = None,
) -> list[int]:
    """Validate and coerce years to a bounded list of ints."""
    if max_year is None:
        max_year = dt.date.today().year

    if not years:
        raise ValueError("year_whitelist must be non-empty")

    bad = [y for y in years if y < min_year or y > max_year]
    if bad:
        raise ValueError(
            f"Invalid years {bad}. Expected range [{min_year}, {max_year}]."
        )

    return years


def _load_payload(path: Path, *, compression: str) -> dict[str, Any]:
    """Load a JSON payload from disk (optionally gzip-compressed)."""
    if compression == "gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if compression == "none":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(
        f"Unsupported compression '{compression}'. Allowed: "
        f"{sorted(ALLOWED_COMPRESSIONS)}"
    )


def _clean_rows(
    rows: list[dict[str, Any]],
    *,
    max_abs_value: float | None,
) -> list[dict[str, Any]]:
    """Clean row dicts:
    - strip whitespace from keys
    - optionally null-out extreme numeric values
    """
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        out: dict[str, Any] = {}

        for k, v in row.items():
            key = str(k).strip()

            if max_abs_value is not None and isinstance(v, (int, float)):
                if abs(v) > max_abs_value:
                    out[key] = None
                    continue

            out[key] = v

        cleaned.append(out)

    return cleaned


def _payload_to_df(
    payload: dict[str, Any],
    *,
    max_abs_value: float | None,
) -> pl.DataFrame:
    """Convert ORATS payload -> Polars DataFrame from payload['data']."""
    rows = payload.get("data", [])
    if not rows:
        return pl.DataFrame()

    cleaned = _clean_rows(rows, max_abs_value=max_abs_value)
    return pl.DataFrame(cleaned)


def _raw_year_dir(raw_root: Path, endpoint: str, year: int) -> Path:
    return raw_root / f"endpoint={endpoint}" / f"year={year}"


def _raw_full_dir(raw_root: Path, endpoint: str, ticker: str) -> Path:
    return raw_root / f"endpoint={endpoint}" / f"underlying={ticker}"


def _intermediate_full_history(
    intermediate_root: Path,
    endpoint: str,
    ticker: str,
) -> Path:
    return (
        intermediate_root
        / f"endpoint={endpoint}"
        / f"underlying={ticker}"
        / "full.parquet"
    )


def _glob_raw_suffix(compression: str) -> str:
    if compression == "gz":
        return "*.json.gz"
    if compression == "none":
        return "*.json"
    raise ValueError(
        f"Unsupported compression '{compression}'. Allowed: "
        f"{sorted(ALLOWED_COMPRESSIONS)}"
    )


# ----------------------------------------------------------------------------
# Extraction handlers (private)
# ----------------------------------------------------------------------------

def _extract_full_history(
    *,
    endpoint: str,
    raw_root: Path,
    intermediate_root: Path,
    tickers: list[str] | None,
    compression: str,
    overwrite: bool,
    parquet_compression: str,
    max_abs_value: float | None,
) -> ExtractApiResult:
    """Extract FULL_HISTORY raw payloads (one file per ticker)."""
    out_paths: list[Path] = []
    failed: list[Path] = []
    n_seen = 0
    n_read = 0
    n_rows = 0

    base = raw_root / f"endpoint={endpoint}"
    if not base.exists():
        raise FileNotFoundError(f"Raw endpoint folder not found: {base}")

    if tickers is None:
        # Infer tickers from directory layout: underlying=<TICKER>
        tickers = sorted(
            p.name.split("=", 1)[1]
            for p in base.glob("underlying=*")
            if p.is_dir()
        )

    logger.info(
        "Extract FULL_HISTORY endpoint=%s tickers=%d",
        endpoint,
        len(tickers),
    )

    suffix = _glob_raw_suffix(compression)

    for ticker in tickers:
        raw_dir = _raw_full_dir(raw_root, endpoint, ticker)
        files = sorted(raw_dir.glob(suffix))
        if not files:
            continue

        # Usually only one file, but we tolerate multiple.
        for fp in files:
            n_seen += 1

            out_path = _intermediate_full_history(
                intermediate_root, 
                endpoint, 
                ticker
            )
            if (not overwrite) and out_path.exists():
                logger.debug(
                    "Skipping existing intermediate endpoint=%s ticker=%s path=%s",
                    endpoint,
                    ticker,
                    out_path,
                )
                continue

            try:
                payload = _load_payload(fp, compression=compression)
                df = _payload_to_df(payload, max_abs_value=max_abs_value)
                n_read += 1

                if df.height == 0:
                    logger.debug(
                        "Empty data endpoint=%s ticker=%s file=%s",
                        endpoint,
                        ticker,
                        fp,
                    )
                    # Still write empty parquet if overwrite or file missing.
                    _ensure_dir(out_path.parent)
                    df.write_parquet(out_path, compression=parquet_compression)
                    out_paths.append(out_path)
                    continue

                n_rows += df.height
                _ensure_dir(out_path.parent)
                df.write_parquet(out_path, compression=parquet_compression)
                out_paths.append(out_path)

            except Exception:
                failed.append(fp)

    return ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.FULL_HISTORY,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        out_paths=out_paths,
        failed_paths=failed,
    )


def _extract_by_trade_date(
    *,
    endpoint: str,
    raw_root: Path,
    intermediate_root: Path,
    tickers: list[str] | None,
    years: list[int],
    compression: str,
    overwrite: bool,
    parquet_compression: str,
    max_abs_value: float | None,
) -> ExtractApiResult:
    """Extract BY_TRADE_DATE raw payloads and write one parquet per ticker.

    Raw is partitioned by year/date chunks (for download/resume).
    Intermediate is ticker-centric across the full history (for joins/usage).
    """
    out_paths: list[Path] = []
    failed: list[Path] = []
    n_seen = 0
    n_read = 0
    n_rows = 0

    suffix = _glob_raw_suffix(compression)

    logger.info(
        "Extract BY_TRADE_DATE endpoint=%s years=%d",
        endpoint,
        len(years),
    )

    # Accumulate across *all* years, then write once per ticker.
    per_ticker: dict[str, list[pl.DataFrame]] = {}

    for year in years:
        year_dir = _raw_year_dir(raw_root, endpoint, year)
        files = sorted(year_dir.glob(suffix))
        if not files:
            continue

        for fp in files:
            n_seen += 1

            try:
                payload = _load_payload(fp, compression=compression)
                df = _payload_to_df(payload, max_abs_value=max_abs_value)
                n_read += 1

                if df.height == 0:
                    continue

                if "ticker" not in df.columns:
                    # If ORATS ever omits ticker, we cannot split reliably.
                    raise KeyError(f"Missing 'ticker' column in {fp}")

                if tickers is not None:
                    df = df.filter(pl.col("ticker").is_in(tickers))
                    if df.height == 0:
                        continue

                for t, g in df.group_by("ticker", maintain_order=True):
                    t_str = str(t)
                    per_ticker.setdefault(t_str, []).append(g)

                n_rows += df.height

            except Exception:
                failed.append(fp)

    # Write one parquet per ticker (full history).
    for t, dfs in per_ticker.items():
        out_path = _intermediate_full_history(
            intermediate_root,
            endpoint,
            t,
        )

        if (not overwrite) and out_path.exists():
            logger.debug(
                "Skipping existing intermediate endpoint=%s ticker=%s path=%s",
                endpoint,
                t,
                out_path,
            )
            continue

        df_all = pl.concat(dfs, how="diagonal_relaxed")
        _ensure_dir(out_path.parent)
        df_all.write_parquet(out_path, compression=parquet_compression)
        out_paths.append(out_path)

    return ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.BY_TRADE_DATE,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        out_paths=out_paths,
        failed_paths=failed,
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def extract(
    *,
    endpoint: str,
    raw_root: str | Path,
    intermediate_root: str | Path,
    tickers: list[str] | None = None,
    year_whitelist: list[int] | list[str] | None = None,
    compression: str = "gz",
    overwrite: bool = False,
    parquet_compression: str = "zstd",
    max_abs_value: float | None = 1e20,
) -> ExtractApiResult:
    """Extract ORATS raw API snapshots into intermediate parquet.

    Parameters
    ----------
    endpoint:
        Endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
    raw_root:
        Root of raw snapshots produced by `orats_downloader_api.download`.
    intermediate_root:
        Root of intermediate parquet output.
    tickers:
        Optional ticker allowlist. If None:
          - FULL_HISTORY: inferred from `underlying=*` folders.
          - BY_TRADE_DATE: keep all tickers present in payloads.
    year_whitelist:
        Required for BY_TRADE_DATE endpoints. Ignored for FULL_HISTORY.
    compression:
        "gz" or "none" (must match how raw snapshots were written).
    overwrite:
        If False (default), skip intermediate files that already exist.
    parquet_compression:
        Parquet compression (e.g. "zstd", "snappy").
    max_abs_value:
        If not None, any numeric value with abs(value) > max_abs_value is
        replaced by None during extraction (helps avoid insane data glitches).

    Returns
    -------
    ExtractApiResult:
        Summary including written files and failures.
    """
    raw_root_p = Path(raw_root)
    interm_root_p = Path(intermediate_root)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    spec = get_endpoint_spec(endpoint)

    if spec.strategy == DownloadStrategy.FULL_HISTORY:
        if year_whitelist is not None:
            logger.warning(
                "year_whitelist is ignored for endpoint=%s (FULL_HISTORY)",
                endpoint,
            )
        return _extract_full_history(
            endpoint=endpoint,
            raw_root=raw_root_p,
            intermediate_root=interm_root_p,
            tickers=tickers,
            compression=compression,
            overwrite=overwrite,
            parquet_compression=parquet_compression,
            max_abs_value=max_abs_value,
        )

    # BY_TRADE_DATE
    if year_whitelist is None:
        raise ValueError(
            "year_whitelist must be provided for BY_TRADE_DATE endpoints"
        )
    years = _validate_years([int(y) for y in year_whitelist])

    return _extract_by_trade_date(
        endpoint=endpoint,
        raw_root=raw_root_p,
        intermediate_root=interm_root_p,
        tickers=tickers,
        years=years,
        compression=compression,
        overwrite=overwrite,
        parquet_compression=parquet_compression,
        max_abs_value=max_abs_value,
    )