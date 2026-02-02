from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from ..io import ALLOWED_COMPRESSIONS, ensure_dir, json_suffix
from volatility_trading.config.orats.api_schemas import get_schema_spec

logger = logging.getLogger(__name__)


def is_non_fatal_extract_error(e: Exception) -> bool:
    """Return True for errors where we should record+continue."""
    return isinstance(
        e,
        (
            OSError,
            gzip.BadGzipFile,
            json.JSONDecodeError,
            UnicodeDecodeError,
            pl.exceptions.PolarsError,
        ),
    )


def load_payload(path: Path, *, compression: str) -> dict[str, Any]:
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


def write_parquet_atomic(
    df: pl.DataFrame,
    path: Path,
    *,
    compression: str,
) -> None:
    """Write parquet atomically using a temp file + rename."""
    ensure_dir(path.parent)
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

        df.write_parquet(tmp_path, compression=compression)

        try:
            with open(tmp_path, mode="rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass

        os.replace(tmp_path, path)

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


def remove_duplicates(
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


def payload_to_df(endpoint: str, payload: dict[str, Any]) -> pl.DataFrame:
    """Convert ORATS payload -> Polars DataFrame from payload['data']."""
    rows = payload.get("data", [])
    if not rows:
        return pl.DataFrame()

    spec = get_schema_spec(endpoint)
    if spec is None:
        return pl.DataFrame(rows, infer_schema_length=None)

    dtypes: dict[str, pl.DataType] = dict(getattr(spec, "vendor_dtypes", {}))
    date_cols: tuple[str, ...] = getattr(spec, "vendor_date_cols", ())
    datetime_cols: tuple[str, ...] = getattr(spec, "vendor_datetime_cols", ())

    df = pl.from_dicts(
        rows,
        schema_overrides=dtypes if dtypes else None,
        infer_schema_length=None,
    )

    if date_cols:
        exprs: list[pl.Expr] = []
        for c in date_cols:
            if c in df.columns:
                exprs.append(
                    pl.col(c).str.strptime(pl.Date, strict=False).alias(c)
                )
        if exprs:
            df = df.with_columns(exprs)

    if datetime_cols:
        exprs2: list[pl.Expr] = []
        dt_type = pl.Datetime(time_zone="UTC")
        for c in datetime_cols:
            if c in df.columns:
                exprs2.append(
                    pl.col(c).str.strptime(dt_type, strict=False).alias(c)
                )
        if exprs2:
            df = df.with_columns(exprs2)

    return df


def safe_rename(df: pl.DataFrame, renames: dict[str, str]) -> pl.DataFrame:
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


def apply_endpoint_schema(endpoint: str, df: pl.DataFrame) -> pl.DataFrame:
    """Apply endpoint-specific schema transforms (rename/keep)."""
    spec = get_schema_spec(endpoint)
    if spec is None or df.is_empty():
        return df

    renames: dict[str, str] = getattr(spec, "renames_vendor_to_canonical", {})
    keep: tuple[str, ...] | None = getattr(spec, "keep_canonical", None)

    df2 = safe_rename(df, renames)

    if keep:
        cols = [c for c in keep if c in df2.columns]
        if cols:
            df2 = df2.select(cols)

    return df2


def glob_raw_suffix(compression: str) -> str:
    """Glob pattern for raw snapshot files for a given compression mode."""
    return f"*{json_suffix(compression)}"