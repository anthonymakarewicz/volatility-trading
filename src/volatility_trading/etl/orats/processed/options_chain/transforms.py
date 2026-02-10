"""Private transformation helpers for the processed options-chain builder."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Dedupe helper
# ----------------------------------------------------------------------------


def dedupe_on_keys(
    lf: pl.LazyFrame,
    *,
    key_common: Sequence[str],
    key_when_opra_present: Sequence[str] | None = None,
    opra_nonnull_cols: Sequence[str] | None = None,
    stable_sort: bool = False,
) -> pl.LazyFrame:
    """Drop exact duplicates using best-available key columns.

    Notes
    -----
    This is a *mechanical* de-duplication helper for processed-building:
    - Rows with nulls in `key_common` are dropped.
    - If OPRA codes are present and you provide `key_when_opra_present` and
      `opra_nonnull_cols`, then:
        * rows where all OPRA cols are non-null are de-duped on
          `key_when_opra_present`
        * rows missing OPRA cols are de-duped on `key_common`

    If you need "latest wins" semantics, you must include a timestamp column
    and sort by it before calling `.unique(keep="last")`.
    """
    schema = lf.collect_schema()
    cols = set(schema.names())

    key_common_eff = [c for c in key_common if c in cols]
    if not key_common_eff:
        raise ValueError(
            f"_dedupe_on_keys: none of key_common columns exist: {list(key_common)}"
        )

    # Drop rows with nulls in the always-required keys.
    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in key_common_eff]))

    def _unique_on(subset: Sequence[str], lf_in: pl.LazyFrame) -> pl.LazyFrame:
        subset_eff = [c for c in subset if c in cols]
        if not subset_eff:
            return lf_in
        # Optional deterministic ordering (can be expensive on large scans).
        if stable_sort:
            lf_in = lf_in.sort(subset_eff)
        return lf_in.unique(subset=subset_eff, maintain_order=True)

    # No OPRA-aware logic requested.
    if not (key_when_opra_present and opra_nonnull_cols):
        return _unique_on(key_common_eff, lf)

    opra_cols_eff = [c for c in opra_nonnull_cols if c in cols]
    if len(opra_cols_eff) != len(opra_nonnull_cols):
        # OPRA columns not available in this scan; fall back to common key.
        return _unique_on(key_common_eff, lf)

    has_opra_expr = pl.all_horizontal([pl.col(c).is_not_null() for c in opra_cols_eff])

    lf_with = _unique_on(key_when_opra_present, lf.filter(has_opra_expr))
    lf_without = _unique_on(key_common_eff, lf.filter(~has_opra_expr))

    return pl.concat([lf_with, lf_without], how="vertical")
