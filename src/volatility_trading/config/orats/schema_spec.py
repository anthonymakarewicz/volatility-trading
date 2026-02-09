"""
Shared ORATS schema spec (API + FTP).

This dataclass is a small "contract" used by ETL steps to:
- cast vendor fields to stable dtypes
- parse vendor date/datetime fields
- rename vendor fields -> canonical snake_case
- select a curated set of canonical columns to keep (intermediate)
- apply two tiers of canonical bounds:
    * bounds_drop_canonical: out-of-bounds => drop row (structural validity)
    * bounds_null_canonical: out-of-bounds => set value to null (row can survive)

Notes
-----
- Bounds here are for preventing absurd values from contaminating downstream
  processing. They are not meant to encode strategy logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class OratsSchemaSpec:
    """Normalization spec for one ORATS dataset (API endpoint or FTP dataset).

    Attributes
    ----------
    vendor_dtypes:
        Mapping vendor/original field name -> Polars dtype to cast to (pre-rename).
    vendor_date_cols:
        Vendor field names to parse/cast as `pl.Date` (pre-rename).
    vendor_datetime_cols:
        Vendor field names to parse/cast as `pl.Datetime` (pre-rename).
    renames_vendor_to_canonical:
        Mapping vendor/original field name -> canonical column name (snake_case).
    keep_canonical:
        Canonical columns to keep in intermediate (post-rename).
    bounds_drop_canonical:
        Optional mapping canonical numeric column -> (lo, hi).
        Rows with values outside bounds are dropped.
    bounds_null_canonical:
        Optional mapping canonical numeric column -> (lo, hi).
        Values outside bounds are set to null.
    """

    vendor_dtypes: dict[str, type[pl.DataType]]
    renames_vendor_to_canonical: dict[str, str]
    keep_canonical: tuple[str, ...]

    vendor_date_cols: tuple[str, ...] = ()
    vendor_datetime_cols: tuple[str, ...] = ()

    bounds_drop_canonical: dict[str, tuple[float, float]] | None = None
    bounds_null_canonical: dict[str, tuple[float, float]] | None = None
