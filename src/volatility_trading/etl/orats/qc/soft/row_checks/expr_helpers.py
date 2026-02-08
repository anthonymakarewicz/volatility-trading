from __future__ import annotations

import polars as pl


def _all_not_null(cols: list[str]) -> pl.Expr:
    """
    Convenience helper:
    returns pl.all_horizontal([pl.col(c).is_not_null() for c in cols])
    """
    return pl.all_horizontal([pl.col(c).is_not_null() for c in cols])


def _apply_required_mask(
    *,
    violation_expr: pl.Expr,
    required_expr: pl.Expr,
    out_col: str,
) -> pl.Expr:
    """
    Apply a required mask to a boolean violation expression:

        when(required).then(violation).otherwise(False)

    plus null safety and alias(out_col).
    """
    return (
        pl.when(required_expr)
        .then(violation_expr)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )
