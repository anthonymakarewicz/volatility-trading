"""
Private transformation helpers used by the ORATS daily-features builder.

This module contains:
- generic logging/counters helpers
- bounds logic (drop vs null)
- de-duplication logic
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from fnmatch import fnmatch

import polars as pl

from .config import (
    DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS,
    DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS_GLOB,
    DAILY_FEATURES_UNITS_STRICT,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Logging / stats helpers
# ----------------------------------------------------------------------------

def count_rows(lf: pl.LazyFrame) -> int:
    """Count rows in a LazyFrame (forces a small collect)."""
    return int(lf.select(pl.len()).collect().item())


def fmt_int(n: int | None) -> str:
    """Format integers with thousands separators for logging."""
    if n is None:
        return "NA"
    return f"{int(n):,}"


def log_before_after(
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
        fmt_int(before),
        fmt_int(after),
        removed_word,
        fmt_int(removed),
        pct,
    )


def log_total_missing(
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
        fmt_int(total),
        missing_word,
        fmt_int(missing),
        pct,
    )


def apply_unit_multipliers(
    lf: pl.LazyFrame,
    *,
    endpoint: str,
    strict: bool | None = None,
) -> pl.LazyFrame:
    """Apply endpoint-specific unit scaling (e.g. percent -> decimal).

    Rules
    -----
    - exact-name multipliers applied first
    - then glob patterns applied (fnmatch) in declared order
    - only applies to columns that exist in lf schema
    - if strict=True: raises if any column matches multiple glob patterns
    """
    strict_eff = DAILY_FEATURES_UNITS_STRICT if strict is None else bool(strict)

    exact_map = DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS.get(endpoint, {})
    glob_rules = DAILY_FEATURES_ENDPOINT_UNIT_MULTIPLIERS_GLOB.get(endpoint, [])

    if not exact_map and not glob_rules:
        return lf

    schema = lf.collect_schema()
    cols = list(schema.names())
    cols_set = set(cols)

    # 1) exact column multipliers
    exprs: list[pl.Expr] = []
    applied_cols: list[str] = []

    for c, mult in exact_map.items():
        if c in cols_set:
            exprs.append((pl.col(c) * float(mult)).alias(c))
            applied_cols.append(c)

    # 2) glob/pattern multipliers
    # track which patterns matched which columns for strict validation
    glob_matches: dict[str, list[str]] = {}  # col -> [pattern, ...]
    glob_mult: dict[str, float] = {}  # col -> mult (first match wins unless strict)

    if glob_rules:
        for c in cols:
            if c in exact_map:
                # exact already defines it; don't also glob-scale it
                continue
            matched_patterns: list[str] = []
            matched_mults: list[float] = []

            for pat, mult in glob_rules:
                if fnmatch(c, pat):
                    matched_patterns.append(pat)
                    matched_mults.append(float(mult))

            if not matched_patterns:
                continue

            glob_matches[c] = matched_patterns

            if strict_eff and len(matched_patterns) > 1:
                # refuse ambiguous scaling
                raise ValueError(
                    f"apply_unit_multipliers(endpoint={endpoint!r}): "
                    f"column {c!r} matched multiple patterns: {matched_patterns}"
                )

            # first match wins (deterministic)
            glob_mult[c] = matched_mults[0]

        for c, mult in glob_mult.items():
            exprs.append((pl.col(c) * float(mult)).alias(c))
            applied_cols.append(c)

    if not exprs:
        return lf

    logger.info(
        "Units: endpoint=%s scaled_cols=%s",
        endpoint,
        applied_cols,
    )
    return lf.with_columns(exprs)


# ----------------------------------------------------------------------------
# Bounds helpers (optional; useful if you add extra endpoints later)
# ----------------------------------------------------------------------------

def apply_bounds_null(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Set out-of-bounds numeric values to null (row survives)."""
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


def apply_bounds_drop(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Drop rows that violate structural bounds (row is removed)."""
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


def count_rows_any_oob(
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

def dedupe_on_keys(
    lf: pl.LazyFrame,
    *,
    keys: Sequence[str],
    stable_sort: bool = False,
) -> pl.LazyFrame:
    """Drop exact duplicates using key columns.

    Notes
    -----
    - Rows with nulls in `keys` are dropped.
    - If you need "latest wins" semantics, sort by a timestamp first and
      then use `.unique(keep="last")`.
    """
    schema = lf.collect_schema()
    cols = set(schema.names())

    keys_eff = [c for c in keys if c in cols]
    if not keys_eff:
        raise ValueError(
            f"dedupe_on_keys: none of key columns exist: {list(keys)}"
        )

    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in keys_eff]))

    if stable_sort:
        lf = lf.sort(keys_eff)

    return lf.unique(subset=keys_eff, maintain_order=True)
