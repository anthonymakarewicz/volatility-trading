"""Filtering steps for processed options-chain panels."""

from __future__ import annotations

import polars as pl

from ...shared.log_fmt import log_before_after
from ...shared.stats import count_rows
from ..types import BuildStats


def apply_filters(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    dte_min: int,
    dte_max: int,
    moneyness_min: float,
    moneyness_max: float,
    collect_stats: bool,
    stats: BuildStats,
) -> pl.LazyFrame:
    """Apply trading-band filters and hard sanity filters."""
    # 8A) Trading band filters
    n_before_trading: int | None = count_rows(lf) if collect_stats else None

    lf = lf.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),
    )

    if collect_stats:
        stats.n_rows_after_trading = count_rows(lf)
        log_before_after(
            label="Trading filters",
            ticker=ticker,
            before=n_before_trading,
            after=stats.n_rows_after_trading,
            removed_word="dropped",
        )

    # 8B) Hard sanity filters
    n_before_hard: int | None = count_rows(lf) if collect_stats else None

    lf = lf.filter(
        pl.col("trade_date") <= pl.col("expiry_date"),
        pl.col("call_ask_price") >= pl.col("call_bid_price"),
        pl.col("put_ask_price") >= pl.col("put_bid_price"),
        ~(
            (pl.col("call_bid_price") == 0)
            & (pl.col("call_ask_price") == 0)
            & (pl.col("call_model_price") == 0)
            & (pl.col("put_bid_price") == 0)
            & (pl.col("put_ask_price") == 0)
            & (pl.col("put_model_price") == 0)
        ),
    )

    if collect_stats:
        stats.n_rows_after_hard = count_rows(lf)
        log_before_after(
            label="Hard sanity",
            ticker=ticker,
            before=n_before_hard,
            after=stats.n_rows_after_hard,
            removed_word="dropped",
        )

    return lf
