# qc/serialization.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl


def df_to_jsonable_records(
    df: pl.DataFrame,
    *,
    max_rows: int | None = None,
    cols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a (small) Polars DataFrame into JSON-safe list[dict].

    - Optionally select a subset of columns.
    - Optionally cap rows via max_rows.
    - Cast temporal columns to Utf8 to avoid json serialization issues.
    - Cast Decimal to Float64 (or string if you prefer) to avoid json issues.

    Intended for attaching samples/examples into QCCheckResult.details.
    """
    if df.height == 0:
        return []

    out = df

    if cols is not None:
        keep = [c for c in cols if c in out.columns]
        if keep:
            out = out.select(keep)

    if max_rows is not None:
        out = out.head(max_rows)

    # Make JSON-safe
    for col, dtype in out.schema.items():
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            out = out.with_columns(pl.col(col).cast(pl.Utf8))
        elif dtype == pl.Decimal:
            out = out.with_columns(pl.col(col).cast(pl.Float64))

    return out.to_dicts()
