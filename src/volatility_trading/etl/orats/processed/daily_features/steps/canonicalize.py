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
    priority_endpoints: Sequence[str] | None = None,
    strict: bool = True,
) -> pl.LazyFrame:
    """Expose canonical (unprefixed) output columns.

    If prefix_cols=True, map '<endpoint>__<col>' into canonical '<col>'.
    If multiple endpoints provide the same canonical name, values are coalesced
    in the order of:
      - priority_endpoints if provided
      - else endpoints

    Notes
    -----
    - Intended to run after join_endpoints_on_spine().
    - If strict=True (default), fails fast if a requested output column cannot
      be produced. If strict=False, missing columns are filled with null.
    """
    if not output_columns:
        raise ValueError("output_columns must be non-empty")

    if priority_endpoints is not None:
        missing = [ep for ep in priority_endpoints if ep not in endpoints]
        if missing:
            raise ValueError(
                "priority_endpoints must be a subset of endpoints; "
                f"missing={missing}"
            )
        order = list(priority_endpoints)
    else:
        order = list(endpoints)

    schema = lf.collect_schema()
    cols = set(schema.names())

    required_keys = ("ticker", "trade_date")
    for k in required_keys:
        if k not in cols:
            raise ValueError(
                f"canonicalize_columns: missing required key column {k!r}"
            )

    exprs: list[pl.Expr] = []
    missing_out: list[str] = []

    for out_col in output_columns:
        # Keys pass through.
        if out_col in required_keys:
            exprs.append(pl.col(out_col))
            continue

        # If not prefixing, we expect the column to exist as-is.
        if not prefix_cols:
            if out_col not in cols:
                missing_out.append(out_col)
                exprs.append(pl.lit(None).alias(out_col))
            else:
                exprs.append(pl.col(out_col))
            continue

        # prefix_cols=True: prefer already-unprefixed if present.
        if out_col in cols:
            exprs.append(pl.col(out_col))
            continue

        # Otherwise look for endpoint-prefixed candidates in the chosen order.
        candidates: list[str] = []
        for ep in order:
            c = f"{ep}__{out_col}"
            if c in cols:
                candidates.append(c)

        if not candidates:
            missing_out.append(out_col)
            exprs.append(pl.lit(None).alias(out_col))
            continue

        if len(candidates) == 1:
            exprs.append(pl.col(candidates[0]).alias(out_col))
            continue

        # Multiple providers -> coalesce in priority order
        exprs.append(
            pl.coalesce([pl.col(c) for c in candidates]).alias(out_col)
        )

    if strict and missing_out:
        raise ValueError(
            "canonicalize_columns: could not produce requested columns: "
            + ", ".join(missing_out)
        )

    # Selecting expressions also drops unused prefixed columns automatically.
    return lf.select(exprs)