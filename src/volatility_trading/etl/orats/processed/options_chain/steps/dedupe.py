# volatility_trading/etl/orats/processed/options_chain/_steps/dedupe.py

from __future__ import annotations

import logging

import polars as pl

from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT

from ...shared.log_fmt import log_before_after
from ...shared.stats import count_rows

from ..types import BuildStats
from ..transforms import dedupe_on_keys


logger = logging.getLogger(__name__)


def filter_preferred_opra_root(
    *,
    lf: pl.LazyFrame,
    ticker: str,
) -> pl.LazyFrame:
    preferred_root = PREFERRED_OPRA_ROOT.get(ticker)
    if preferred_root is None:
        return lf

    return lf.filter(
        # For older years, call_opra is null / absent; keep those rows.
        pl.col("call_opra").is_null()
        | pl.col("call_opra").str.starts_with(preferred_root)
    )


def dedupe_options_chain(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    collect_stats: bool,
    stats: BuildStats,
) -> pl.LazyFrame:
    logger.info(
        "Applying key null-checks and de-duplication for ticker=%s",
        ticker
    )

    n_before: int | None = count_rows(lf) if collect_stats else None

    lf = dedupe_on_keys(
        lf,
        key_common=["ticker", "trade_date", "expiry_date", "strike"],
        key_when_opra_present=[
            "ticker", "trade_date", "strike", "call_opra", "put_opra"
        ],
        opra_nonnull_cols=["call_opra", "put_opra"],
        stable_sort=False,
    )

    if collect_stats:
        stats.n_rows_after_dedupe = count_rows(lf)
        log_before_after(
            label="Dedupe (options chain)",
            ticker=ticker,
            before=n_before,
            after=stats.n_rows_after_dedupe,
            removed_word="removed",
        )

    return lf