"""volatility_trading.etl.orats.processed.options_chain_transforms

Private transformation helpers used by the ORATS options-chain builder.

This module contains:
- generic logging/counters helpers
- bounds logic (drop vs null)
- de-duplication logic
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Logging / stats helpers
# ----------------------------------------------------------------------------

def _count_rows(lf: pl.LazyFrame) -> int:
    """Count rows in a LazyFrame (forces a small collect)."""
    return int(lf.select(pl.len()).collect().item())


def _fmt_int(n: int | None) -> str:
    """Format integers with thousands separators for logging."""
    if n is None:
        return "NA"
    return f"{int(n):,}"


def _log_before_after(
    *,
    label: str,
    ticker: str,
    before: int | None,
    after: int | None,
    removed_word: str = "removed",
) -> None:
    """Log a standard before/after counter line with percent removed."""
    if before is None or after is None:
        return
    removed = int(before) - int(after)
    pct = (100.0 * removed / before) if before else 0.0
    logger.info(
        "%s ticker=%s before=%s after=%s %s=%s (%.2f%%)",
        label,
        ticker,
        _fmt_int(before),
        _fmt_int(after),
        removed_word,
        _fmt_int(removed),
        pct,
    )


def _log_total_missing(
    *,
    label: str,
    ticker: str,
    total: int | None,
    missing: int | None,
    total_word: str = "rows",
    missing_word: str = "missing",
) -> None:
    """Log a standard total/missing counter line with percent missing."""
    if total is None or missing is None:
        return
    pct = (100.0 * int(missing) / int(total)) if total else 0.0
    logger.info(
        "%s ticker=%s %s=%s %s=%s (%.2f%%)",
        label,
        ticker,
        total_word,
        _fmt_int(total),
        missing_word,
        _fmt_int(missing),
        pct,
    )


# ----------------------------------------------------------------------------
# Bounds helpers
# ----------------------------------------------------------------------------

def _apply_bounds_null(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Set out-of-bounds numeric values to null (row survives).

    Notes
    -----
    - Missing columns are ignored.
    - Nulls remain null.
    """
    if not bounds:
        return lf

    schema = lf.collect_schema()
    cols = set(schema.names())

    exprs: list[pl.Expr] = []
    for c, (lo, hi) in bounds.items():
        if c not in cols:
            continue
        exprs.append(
            pl.when(pl.col(c).is_null())
            .then(pl.col(c))
            .when(pl.col(c).is_between(lo, hi))
            .then(pl.col(c))
            .otherwise(None)
            .alias(c)
        )

    return lf.with_columns(exprs) if exprs else lf


def _apply_bounds_drop(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Drop rows that violate structural bounds (row is removed).

    Notes
    -----
    - Missing columns are ignored.
    - For DROP bounds, nulls are treated as invalid (row is dropped).
    """
    if not bounds:
        return lf

    schema = lf.collect_schema()
    cols = set(schema.names())

    filters: list[pl.Expr] = []
    for c, (lo, hi) in bounds.items():
        if c not in cols:
            continue
        filters.append(pl.col(c).is_not_null() & pl.col(c).is_between(lo, hi))

    return lf.filter(pl.all_horizontal(filters)) if filters else lf


def _count_rows_any_oob(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> tuple[int | None, int | None]:
    """Best-effort stats for bounds-null: (rows_total, rows_with_any_oob)."""
    if not bounds:
        return None, None

    schema = lf.collect_schema()
    cols = set(schema.names())

    oob_exprs: list[pl.Expr] = []
    for c, (lo, hi) in bounds.items():
        if c not in cols:
            continue
        oob_exprs.append(pl.col(c).is_not_null() & ~pl.col(c).is_between(lo, hi))

    if not oob_exprs:
        return None, None

    try:
        out = (
            lf.select(
                pl.len().alias("_n"),
                pl.any_horizontal(oob_exprs).sum().alias("_rows_oob"),
            )
            .collect()
            .row(0)
        )
        return int(out[0]), int(out[1])
    except Exception:
        logger.debug("Bounds-null stats failed", exc_info=True)
        return None, None


# ----------------------------------------------------------------------------
# Dedupe helper
# ----------------------------------------------------------------------------

def _dedupe_on_keys(
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
            f"_dedupe_on_keys: none of key_common "
            f"columns exist: {list(key_common)}"
        )

    # Drop rows with nulls in the always-required keys.
    lf = lf.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in key_common_eff])
    )

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

    has_opra_expr = pl.all_horizontal(
        [pl.col(c).is_not_null() for c in opra_cols_eff]
    )

    lf_with = _unique_on(key_when_opra_present, lf.filter(has_opra_expr))
    lf_without = _unique_on(key_common_eff, lf.filter(~has_opra_expr))

    return pl.concat([lf_with, lf_without], how="vertical")