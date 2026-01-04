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

from .orats_api_endpoints import DownloadStrategy, get_endpoint_spec
from .orats_api_io import (
    ALLOWED_COMPRESSIONS,
    ensure_dir,
    intermediate_full_history,
    json_suffix,
    raw_by_trade_date_dir,
    raw_full_history_dir,
    validate_years,
)
from volatility_trading.config.orats_api_schemas import get_schema_spec

logger = logging.getLogger(__name__)

# TODO: Add logger.exption or logger.error(exc_info=True)

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


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean row dicts.

    - Strip whitespace from keys.
    """
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        out: dict[str, Any] = {}
        for k, v in row.items():
            out[str(k).strip()] = v
        cleaned.append(out)

    return cleaned


def _payload_to_df(payload: dict[str, Any]) -> pl.DataFrame:
    """Convert ORATS payload -> Polars DataFrame from payload['data']."""
    rows = payload.get("data", [])
    if not rows:
        return pl.DataFrame()

    cleaned = _clean_rows(rows)
    return pl.DataFrame(cleaned)


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


def _cast_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    """Best-effort casting, with ISO date/datetime parsing when requested."""

    # Polars needs explicit parsing for date/datetime from strings.
    is_date = dtype == pl.Date
    is_dt = getattr(dtype, "__class__", None) is not None and (
        dtype == pl.Datetime or dtype.__class__.__name__ == "Datetime"
    )

    if is_date:
        return pl.col(col).str.strptime(pl.Date, strict=False).alias(col)

    if is_dt:
        return pl.col(col).str.strptime(dtype, strict=False).alias(col)

    return pl.col(col).cast(dtype, strict=False).alias(col)


def _apply_endpoint_schema(
    endpoint: str,
    df: pl.DataFrame,
    *,
    use_bounds: bool,
) -> pl.DataFrame:
    """Apply endpoint-specific schema transforms (rename/cast/bounds/keep)."""
    spec = get_schema_spec(endpoint)
    if spec is None or df.is_empty():
        return df

    renames: dict[str, str] = getattr(spec, "renames", {})
    dtypes: dict[str, pl.DataType] = getattr(spec, "dtypes", {})
    keep: tuple[str, ...] | None = getattr(spec, "keep", None)
    date_cols: tuple[str, ...] = getattr(spec, "date_cols", ())
    datetime_cols: tuple[str, ...] = getattr(spec, "datetime_cols", ())
    bounds: dict[str, tuple[float, float]] | None = getattr(spec, "bounds", None)

    # Allow schemas to specify date/datetime parsing without repeating dtypes.
    if date_cols:
        for c in date_cols:
            dtypes.setdefault(c, pl.Date)

    if datetime_cols:
        for c in datetime_cols:
            dtypes.setdefault(c, pl.Datetime)

    df2 = _safe_rename(df, renames)

    if dtypes:
        exprs: list[pl.Expr] = []
        for c, dt_ in dtypes.items():
            if c in df2.columns:
                exprs.append(_cast_expr(c, dt_))
        if exprs:
            df2 = df2.with_columns(exprs)

    if use_bounds and bounds:
        exprs2: list[pl.Expr] = []
        schema = df2.schema

        for c, b in bounds.items():
            if c not in df2.columns:
                continue
            if not isinstance(b, (tuple, list)) or len(b) != 2:
                continue

            lo, hi = b
            dt_ = schema.get(c)

            expr = (
                pl.when((pl.col(c) >= lo) & (pl.col(c) <= hi))
                .then(pl.col(c))
                .otherwise(pl.lit(None))
            )

            # Preserve the original dtype when possible.
            if dt_ is not None:
                expr = expr.cast(dt_, strict=False)

            exprs2.append(expr.alias(c))

        if exprs2:
            df2 = df2.with_columns(exprs2)

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
    use_bounds: bool,
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
            df = _payload_to_df(payload)
            df = _apply_endpoint_schema(endpoint, df, use_bounds=use_bounds)
            n_read += 1

            if df.height == 0:
                logger.debug(
                    "Empty data endpoint=%s ticker=%s file=%s",
                    endpoint,
                    ticker,
                    fp,
                )
                # Still write empty parquet if overwrite or file missing.
                _write_parquet_atomic(df, out_path, compression=parquet_compression)
                out_paths.append(out_path)
                continue

            n_rows += df.height
            _write_parquet_atomic(df, out_path, compression=parquet_compression)
            out_paths.append(out_path)

        except Exception as e:
            failed.append(fp)
            logger.debug(
                "Failed to read raw file=%s endpoint=%s ticker=%s err=%r",
                fp,
                endpoint,
                ticker,
                e,
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
        "Finished extract FULL_HISTORY endpoint=%s seen=%d read=%d written=%d "
        "failed=%d rows=%d duration_s=%.2f",
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
    use_bounds: bool,
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
        year_dir = raw_by_trade_date_dir(raw_root, endpoint, year)
        files = sorted(year_dir.glob(suffix))
        if not files:
            continue

        for fp in files:
            n_seen += 1

            try:
                payload = _load_payload(fp, compression=compression)
                df = _payload_to_df(payload)
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
                logger.debug("Failed to read raw file=%s err=%r", fp, e)

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
        df_all = _apply_endpoint_schema(endpoint, df_all, use_bounds=use_bounds)
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
        "Finished extract BY_TRADE_DATE endpoint=%s years=%d seen=%d read=%d "
        "written=%d failed=%d rows=%d duration_s=%.2f",
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
    use_bounds: bool = True,
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
    use_bounds:
        If True (default), apply endpoint-specific bounds from the schema spec.
        Values outside bounds are set to null.

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

    if not use_bounds:
        logger.warning(
            "Bounds filtering disabled (use_bounds=False) endpoint=%s",
            endpoint,
        )

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
            use_bounds=use_bounds,
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
        use_bounds=use_bounds,
    )