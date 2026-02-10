"""De-duplication step for processed daily-features endpoint panels."""

from __future__ import annotations

import logging

import polars as pl

from ...shared.log_fmt import log_before_after
from ...shared.stats import count_rows
from ..transforms import dedupe_on_keys

logger = logging.getLogger(__name__)


def dedupe_endpoint(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    endpoint: str,
    collect_stats: bool,
    stats_after_dedupe_by_endpoint: dict[str, int] | None,
) -> pl.LazyFrame:
    """De-dupe an endpoint panel on (ticker, trade_date)."""
    n_before: int | None = count_rows(lf) if collect_stats else None

    lf = dedupe_on_keys(
        lf,
        keys=["ticker", "trade_date"],
        stable_sort=False,
    )

    if collect_stats and stats_after_dedupe_by_endpoint is not None:
        n_after = count_rows(lf)
        stats_after_dedupe_by_endpoint[endpoint] = n_after
        log_before_after(
            label=f"Dedupe ({endpoint})",
            ticker=ticker,
            before=n_before,
            after=n_after,
            removed_word="removed",
        )

    return lf
