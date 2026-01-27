# qc/soft/dataset_checks/utils.py
from __future__ import annotations

from typing import Any

import polars as pl


def _make_examples_json_safe(df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a small Polars DataFrame into a JSON-safe list[dict].

    Polars can hold Date/Datetime/Time types that json.dump can't serialize by
    default; we cast temporals to Utf8.
    """
    if df.height == 0:
        return []

    out = df
    for col, dtype in out.schema.items():
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            out = out.with_columns(pl.col(col).cast(pl.Utf8))

    return out.to_dicts()