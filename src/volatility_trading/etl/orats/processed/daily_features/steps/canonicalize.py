from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)


def canonicalize_columns(
    *,
    lf: pl.LazyFrame,
    endpoints: Sequence[str],
    output_columns: Sequence[str],
    prefix_cols: bool,
) -> pl.LazyFrame:
    """Return a LazyFrame exposing canonical (unprefixed) output columns.

    If prefix_cols=True, this will map '<endpoint>__<col>' columns into the
    canonical '<col>' names. If multiple endpoints provide the same canonical
    name, values are coalesced in the order of `endpoints`.

    Notes
    -----
    - This is intended to run after join_endpoints_on_spine().
    - Fails fast if a requested output column cannot be produced.
    """
    if not output_columns:
        raise ValueError("output_columns must be non-empty")

    schema = lf.collect_schema()
    cols = set(schema.names())

    required_keys = ("ticker", "trade_date")
    for k in required_keys:
        if k not in cols:
            raise ValueError(
                f"canonicalize_columns: missing required key column {k!r}"
            )

    # Build expressions for all requested columns.
    exprs: list[pl.Expr] = []
    missing: list[str] = []

    for out_col in output_columns:
        # Keys pass through.
        if out_col in required_keys:
            exprs.append(pl.col(out_col))
            continue

        # If not prefixing, we expect the column to exist as-is.
        if not prefix_cols:
            if out_col not in cols:
                missing.append(out_col)
                continue
            exprs.append(pl.col(out_col))
            continue

        # prefix_cols=True: prefer already-unprefixed if present.
        if out_col in cols:
            exprs.append(pl.col(out_col))
            continue

        # Otherwise look for endpoint-prefixed candidates.
        candidates: list[str] = []
        for ep in endpoints:
            c = f"{ep}__{out_col}"
            if c in cols:
                candidates.append(c)

        if not candidates:
            missing.append(out_col)
            continue

        if len(candidates) == 1:
            exprs.append(pl.col(candidates[0]).alias(out_col))
            continue

        # Multiple providers: coalesce in endpoints order.
        exprs.append(
            pl.coalesce([pl.col(c) for c in candidates]).alias(out_col)
        )

    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "canonicalize_columns: could not produce requested columns: "
            f"{missing_str}"
        )

    # Selecting expressions also drops unused prefixed columns automatically.
    return lf.select(exprs)