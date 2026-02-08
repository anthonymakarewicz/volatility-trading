# etl/orats/processed/_shared/bounds.py
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


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
