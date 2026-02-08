from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import polars as pl

from volatility_trading.config.orats.ftp_schemas import STRIKES_SCHEMA_SPEC as spec

logger = logging.getLogger(__name__)

DATE_FMT: str = "%m/%d/%Y"


def _normalize_strikes_vendor_df(df: pl.DataFrame) -> pl.DataFrame:
    """Parse vendor date columns and rename vendor columns to canonical."""
    if df.is_empty():
        return df

    # 1) Parse vendor date/datetime columns (before rename)
    exprs: list[pl.Expr] = []

    # Dates (vendor format is typically MM/DD/YYYY)
    for c in spec.vendor_date_cols:
        if c in df.columns:
            exprs.append(
                pl.col(c).str.strptime(pl.Date, format=DATE_FMT, strict=False).alias(c)
            )

    # Datetimes (if ever provided by the vendor). We keep this generic.
    for c in spec.vendor_datetime_cols:
        if c in df.columns:
            exprs.append(pl.col(c).str.strptime(pl.Datetime, strict=False).alias(c))

    if exprs:
        df = df.with_columns(exprs)

    # 2) Rename vendor -> canonical (only if present)
    mapping = {
        src: dst
        for src, dst in spec.renames_vendor_to_canonical.items()
        if src in df.columns and dst and src != dst
    }
    if mapping:
        df = df.rename(mapping)

    # 3) Keep only canonical columns requested by the schema (best-effort)
    if getattr(spec, "keep_canonical", None):
        keep = [c for c in spec.keep_canonical if c in df.columns]
        if keep:
            df = df.select(keep)

    return df


def read_orats_zip_to_polars(zip_path: Path) -> pl.DataFrame:
    """Read one ORATS strikes ZIP (containing a CSV) into a normalized DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # keep only actual CSV files (ignore directories)
        csv_candidates = [
            name
            for name in names
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]

        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV file found inside {zip_path.name}; entries={names}"
            )

        csv_name = csv_candidates[0]

        with zf.open(csv_name) as f:
            df = pl.read_csv(
                f,
                schema_overrides=spec.vendor_dtypes,
                null_values=["NULL"],
            )
            df = _normalize_strikes_vendor_df(df)

    return df
