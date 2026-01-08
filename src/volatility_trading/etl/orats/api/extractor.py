"""ORATS API extractor (raw JSON snapshots -> intermediate parquet).

Reads raw snapshots written by `orats_downloader_api.py` and writes
ticker-centric parquet files under:
  intermediate_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet

Raw layout depends on endpoint strategy (FULL_HISTORY vs BY_TRADE_DATE).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .endpoints import DownloadStrategy, get_endpoint_spec
from .io import (
    ALLOWED_COMPRESSIONS,
    ensure_dir,
    intermediate_full_history,
    json_suffix,
    raw_by_trade_date_dir,
    raw_full_history_dir,
    validate_years,
)
from volatility_trading.config.orats.api_schemas import get_schema_spec

logger = logging.getLogger(__name__)


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
    n_out_files: int
    duration_s: float
    out_paths: list[Path]
    failed_paths: list[Path]


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _is_non_fatal_extract_error(e: Exception) -> bool:
    """Return True for errors where we should record+continue.

    These are typically file-level corruption/IO/parse issues. Everything else
    is treated as fatal and re-raised.
    """
    return isinstance(
        e, (
            OSError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
            pl.exceptions.PolarsError,
        ),
    )

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


def _write_parquet_atomic(
    df: pl.DataFrame,
    path: Path,
    *,
    compression: str,
) -> None:
    """Write parquet atomically using a temp file + rename.

    Writes to a unique temp file in the same directory (same filesystem) and
    then swaps it into place with `os.replace`, which is atomic on POSIX.

    We also attempt `os.fsync` on the temp file and its parent directory
    (best-effort) to improve durability against crashes/power loss.
    """
    ensure_dir(path.parent)
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

        # Polars writes the parquet file.
        df.write_parquet(tmp_path, compression=compression)

        # Best-effort fsync of file contents.
        try:
            with open(tmp_path, mode="rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass

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


def _remove_duplicates(
    df: pl.DataFrame,
    *,
    endpoint: str,
    context: str,
) -> pl.DataFrame:
    """Remove exact duplicate rows and log if row count changes."""
    if df.is_empty():
        return df

    n0 = df.height
    df2 = df.unique(maintain_order=True)
    n1 = df2.height

    if n1 != n0:
        logger.warning(
            "Dropped duplicate rows endpoint=%s %s "
            "before=%d after=%d dropped=%d",
            endpoint,
            context,
            n0,
            n1,
            n0 - n1,
        )

    return df2


def _payload_to_df(endpoint: str, payload: dict[str, Any]) -> pl.DataFrame:
    """Convert ORATS payload -> Polars DataFrame from payload['data']."""
    rows = payload.get("data", [])
    if not rows:
        return pl.DataFrame()

    spec = get_schema_spec(endpoint)
    if spec is None:
        # Fall back to full-row inference.
        return pl.DataFrame(rows, infer_schema_length=None)

    dtypes: dict[str, pl.DataType] = dict(getattr(spec, "vendor_dtypes", {}))
    date_cols: tuple[str, ...] = getattr(spec, "vendor_date_cols", ())
    datetime_cols: tuple[str, ...] = getattr(spec, "vendor_datetime_cols", ())

    # Apply vendor-name dtypes at construction time.
    df = pl.from_dicts(
        rows,
        schema_overrides=dtypes if dtypes else None,
        infer_schema_length=None,
    )

    # Parse declared date columns (vendor/original names).
    if date_cols:
        exprs: list[pl.Expr] = []
        for c in date_cols:
            if c in df.columns:
                exprs.append(
                    pl.col(c)
                    .str.strptime(pl.Date, strict=False)
                    .alias(c)
                )
        if exprs:
            df = df.with_columns(exprs)

    # Parse declared datetime columns (vendor/original names).
    if datetime_cols:
        exprs2: list[pl.Expr] = []
        dt_type = pl.Datetime(time_zone="UTC")
        for c in datetime_cols:
            if c in df.columns:
                exprs2.append(
                    pl.col(c)
                    .str.strptime(dt_type, strict=False)
                    .alias(c)
                )
        if exprs2:
            df = df.with_columns(exprs2)

    return df


def _safe_rename(df: pl.DataFrame, renames: dict[str, str]) -> pl.DataFrame:
    """Rename only columns that exist (avoid hard failures on missing cols)."""
    if not renames or df.is_empty():
        return df

    mapping = {
        src: dst
        for src, dst in renames.items()
        if src in df.columns and dst and src != dst
    }
    if not mapping:
        return df

    return df.rename(mapping)


def _apply_endpoint_schema(
    endpoint: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply endpoint-specific schema transforms (rename/keep)."""
    spec = get_schema_spec(endpoint)
    if spec is None or df.is_empty():
        return df

    renames: dict[str, str] = getattr(spec, "renames_vendor_to_canonical", {})
    keep: tuple[str, ...] | None = getattr(spec, "keep_canonical", None)

    df2 = _safe_rename(df, renames)

    if keep:
        cols = [c for c in keep if c in df2.columns]
        if cols:
            df2 = df2.select(cols)

    return df2


def _glob_raw_suffix(compression: str) -> str:
    """Glob pattern for raw snapshot files for a given compression mode."""
    return f"*{json_suffix(compression)}"


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
) -> ExtractApiResult:
    """Extract FULL_HISTORY raw payloads (one file per ticker)."""
    t0 = time.perf_counter()
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

    for ticker in tickers:
        raw_dir = raw_full_history_dir(raw_root, endpoint, ticker)
        fp = raw_dir / f"data{json_suffix(compression)}"

        # Contract: FULL_HISTORY raw is exactly one file per ticker.
        if not fp.exists():
            logger.debug(
                "Missing raw snapshot endpoint=%s ticker=%s expected=%s",
                endpoint,
                ticker,
                fp,
            )
            continue

        # Best-effort warning if both compressed/uncompressed snapshots exist.
        other = raw_dir / (
            "data.json" if compression == "gz" else "data.json.gz"
        )
        if other.exists():
            logger.warning(
                "Multiple raw snapshots found. Using %s and ignoring %s",
                fp,
                other,
            )

        n_seen += 1
        out_path = intermediate_full_history(
            intermediate_root,
            endpoint,
            ticker,
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
            df = _payload_to_df(endpoint, payload)
            df = _apply_endpoint_schema(endpoint, df)
            df = _remove_duplicates(
                df, 
                endpoint=endpoint, 
                context=f"ticker={ticker}"
            )
            n_read += 1

            if df.height == 0:
                logger.debug(
                    "Empty data endpoint=%s ticker=%s file=%s",
                    endpoint,
                    ticker,
                    fp,
                )
                # Still write empty parquet if overwrite or file missing.
                _write_parquet_atomic(
                    df, 
                    out_path, 
                    compression=parquet_compression
                )
                out_paths.append(out_path)
                continue

            n_rows += df.height
            _write_parquet_atomic(df, out_path, compression=parquet_compression)
            out_paths.append(out_path)

        except Exception as e:
            failed.append(fp)

            if _is_non_fatal_extract_error(e):
                logger.exception(
                    "Non-fatal extract error endpoint=%s ticker=%s file=%s",
                    endpoint,
                    ticker,
                    fp,
                )
                continue

            logger.exception(
                "Fatal extract error endpoint=%s ticker=%s file=%s",
                endpoint,
                ticker,
                fp,
            )
            raise

        if (n_seen % 5) == 0:
            logger.info(
                "Progress FULL_HISTORY endpoint=%s seen=%d "
                "read=%d written=%d failed=%d rows=%d",
                endpoint,
                n_seen,
                n_read,
                len(out_paths),
                len(failed),
                n_rows,
            )

    result = ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.FULL_HISTORY,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        n_out_files=len(out_paths),
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed,
    )

    logger.info(
        "Finished extract FULL_HISTORY endpoint=%s seen=%d read=%d "
        "written=%d failed=%d rows=%d duration_s=%.2f",
        result.endpoint,
        result.n_raw_files_seen,
        result.n_raw_files_read,
        result.n_out_files,
        result.n_failed,
        result.n_rows_total,
        result.duration_s,
    )

    return result


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
) -> ExtractApiResult:
    """
    Extract BY_TRADE_DATE raw payloads and write one parquet per ticker.

    Raw is partitioned by year/date chunks (for download/resume).
    Intermediate is ticker-centric across the full history (for joins/usage).
    """
    t0 = time.perf_counter()
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
        year_t0 = time.perf_counter()
        year_dir = raw_by_trade_date_dir(raw_root, endpoint, year)
        files = sorted(year_dir.glob(suffix))
        if not files:
            logger.debug(
                "No raw files for endpoint=%s year=%d dir=%s",
                endpoint,
                year,
                year_dir,
            )
            continue

        logger.info(
            "Year %d: found %d raw files for endpoint=%s",
            year,
            len(files),
            endpoint,
        )

        for fp in files:
            n_seen += 1

            try:
                payload = _load_payload(fp, compression=compression)
                df = _payload_to_df(endpoint, payload)
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

                for key, g in df.group_by("ticker", maintain_order=True):
                    # Polars yields the group key as a tuple even for 1 column.
                    if isinstance(key, tuple):
                        t_val = key[0]
                    else:
                        t_val = key
                    t_str = str(t_val)
                    per_ticker.setdefault(t_str, []).append(g)

                n_rows += df.height

            except Exception as e:
                failed.append(fp)

                if _is_non_fatal_extract_error(e):
                    logger.exception(
                        "Non-fatal extract error endpoint=%s file=%s",
                        endpoint,
                        fp,
                    )
                    continue

                logger.exception(
                    "Fatal extract error endpoint=%s file=%s",
                    endpoint,
                    fp,
                )
                raise

        logger.info(
            "Year %d done endpoint=%s files=%d seen=%d read=%d "
            "failed=%d rows=%d tickers=%d duration_s=%.2f",
            year,
            endpoint,
            len(files),
            n_seen,
            n_read,
            len(failed),
            n_rows,
            len(per_ticker),
            time.perf_counter() - year_t0,
        )

    # Write one parquet per ticker (full history).
    for t, dfs in per_ticker.items():
        out_path = intermediate_full_history(
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
        df_all = _apply_endpoint_schema(endpoint, df_all)
        df_all = _remove_duplicates(df_all, endpoint=endpoint, context=f"ticker={t}")
        _write_parquet_atomic(df_all, out_path, compression=parquet_compression)
        out_paths.append(out_path)

    result = ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.BY_TRADE_DATE,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        n_out_files=len(out_paths),
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed,
    )

    logger.info(
        "Finished extract BY_TRADE_DATE endpoint=%s years=%d seen=%d "
        "read=%d written=%d failed=%d rows=%d duration_s=%.2f",
        result.endpoint,
        len(years),
        result.n_raw_files_seen,
        result.n_raw_files_read,
        result.n_out_files,
        result.n_failed,
        result.n_rows_total,
        result.duration_s,
    )

    return result


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def extract(
    *,
    endpoint: str,
    raw_root: str | Path,
    intermediate_root: str | Path,
    tickers: Iterable[str] | None = None,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    compression: str = "gz",
    overwrite: bool = False,
    parquet_compression: str = "zstd",
) -> ExtractApiResult:
    """
    Extract ORATS raw API snapshots into intermediate parquet.

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
    
    tickers_clean: list[str] | None
    if tickers is None:
        tickers_clean = None
    else:
        tickers_clean = [
            str(t).strip()
            for t in tickers
            if t is not None and str(t).strip()
        ]
        if not tickers_clean:
            raise ValueError("tickers is passed but none of them is valid")

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
            tickers=tickers_clean,
            compression=compression,
            overwrite=overwrite,
            parquet_compression=parquet_compression,
        )

    # BY_TRADE_DATE
    if year_whitelist is None:
        raise ValueError(
            "year_whitelist must be provided for BY_TRADE_DATE endpoints"
        )
    years = validate_years(year_whitelist)

    return _extract_by_trade_date(
        endpoint=endpoint,
        raw_root=raw_root_p,
        intermediate_root=interm_root_p,
        tickers=tickers_clean,
        years=years,
        compression=compression,
        overwrite=overwrite,
        parquet_compression=parquet_compression,
    )