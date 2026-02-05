# volatility_trading/etl/orats/processed/options_chain/_steps/features.py

from __future__ import annotations

import logging

import polars as pl

from volatility_trading.config.orats.ftp_schemas import STRIKES_SCHEMA_SPEC as spec

from ...shared.bounds import (
    count_rows_any_oob,
    apply_bounds_null,
    apply_bounds_drop,

)
from ...shared.log_fmt import (
    log_before_after,
    log_total_missing,
)
from ...shared.stats import count_rows

logger = logging.getLogger(__name__)


def apply_bounds(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    collect_stats: bool,
) -> pl.LazyFrame:
    # 6A) Bounds NA replacement (NULL)
    bounds_null = getattr(spec, "bounds_null_canonical", None)

    n_total: int | None = None
    if collect_stats:
        n_total, n_rows_oob = count_rows_any_oob(lf, bounds=bounds_null)
        if n_total is not None and n_rows_oob is not None:
            log_total_missing(
                label="Bounds null",
                ticker=ticker,
                total=n_total,
                missing=n_rows_oob,
                total_word="rows",
                missing_word="rows_oob",
            )

    lf = apply_bounds_null(lf, bounds=bounds_null)

    # 6B) Bounds filters (DROP)
    bounds_drop = getattr(spec, "bounds_drop_canonical", None)
    # Reuse the total row count from bounds-null stats when available.
    n_before_drop: int | None
    if collect_stats:
        n_before_drop = n_total if n_total is not None else count_rows(lf)
    else:
        n_before_drop = None

    lf = apply_bounds_drop(lf, bounds=bounds_drop)

    if collect_stats:
        n_after_drop = count_rows(lf)
        log_before_after(
            label="Bounds drop",
            ticker=ticker,
            before=n_before_drop,
            after=n_after_drop,
            removed_word="dropped",
        )

    return lf