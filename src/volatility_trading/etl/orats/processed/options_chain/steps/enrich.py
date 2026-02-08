# volatility_trading/etl/orats/processed/options_chain/_steps/enrich.py

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ...shared.io import scan_endpoint_intermediate
from ...shared.log_fmt import (
    fmt_int,
    log_before_after,
    log_total_missing,
)
from ...shared.stats import count_rows
from ..transforms import dedupe_on_keys
from ..types import BuildStats

logger = logging.getLogger(__name__)


def merge_dividend_yield(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    monies_implied_inter_root: Path | None,
    merge_dividend_yield: bool,
    collect_stats: bool,
    stats: BuildStats,
) -> pl.LazyFrame:
    if not merge_dividend_yield or monies_implied_inter_root is None:
        return lf

    logger.info("Merging dividend yield from monies_implied for ticker=%s", ticker)

    lf_yield = scan_endpoint_intermediate(
        inter_api_root=monies_implied_inter_root,
        ticker=ticker,
        endpoint="monies_implied",
    )

    if collect_stats:
        lf_yield = lf_yield.cache()
        stats.n_rows_yield_input = count_rows(lf_yield)
        logger.info(
            "Input rows (monies_implied intermediate) ticker=%s rows=%s",
            ticker,
            fmt_int(stats.n_rows_yield_input),
        )

    lf_yield = dedupe_on_keys(
        lf_yield,
        key_common=["ticker", "trade_date", "expiry_date"],
        stable_sort=False,
    )

    if collect_stats:
        stats.n_rows_yield_after_dedupe = count_rows(lf_yield)
        log_before_after(
            label="Dedupe (monies_implied)",
            ticker=ticker,
            before=stats.n_rows_yield_input,
            after=stats.n_rows_yield_after_dedupe,
            removed_word="removed",
        )

    # monies_implied is expiry-specific, so join on (ticker, trade_date, expiry_date)
    lf_yield = lf_yield.select(
        ["ticker", "trade_date", "expiry_date", "yield_rate"]
    ).rename({"yield_rate": "_dividend_yield_api"})

    lf = lf.join(lf_yield, on=["ticker", "trade_date", "expiry_date"], how="left")

    if collect_stats:
        try:
            out = (
                lf.select(
                    pl.len().alias("_n"),
                    pl.col("_dividend_yield_api").is_null().sum().alias("_miss"),
                )
                .collect()
                .row(0)
            )
            n_total_join = int(out[0])
            n_miss = int(out[1])
            stats.n_rows_join_missing_yield = n_miss
            log_total_missing(
                label="Yield join",
                ticker=ticker,
                total=n_total_join,
                missing=n_miss,
                total_word="rows",
                missing_word="missing",
            )
        except Exception:
            logger.debug(
                "Yield-join stats failed for ticker=%s",
                ticker,
                exc_info=True,
            )

    # Replace existing dividend_yield
    lf = lf.with_columns(
        pl.coalesce([pl.col("_dividend_yield_api"), pl.col("dividend_yield")]).alias(
            "dividend_yield"
        )
    ).drop(["_dividend_yield_api"])

    return lf


def unify_spot_price(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Unify spot_price across index vs stock/ETF conventions."""
    return lf.with_columns(
        spot_price=pl.when(
            pl.col("spot_price").is_not_null() & (pl.col("spot_price") > 0)
        )
        .then(pl.col("spot_price"))
        .otherwise(pl.col("underlying_price"))
    )
