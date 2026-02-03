"""""volatility_trading.etl.orats.processed.options_chain._steps

Private pipeline steps for building the processed ORATS options chain.

Each step is pure (returns a LazyFrame) and is designed to be orchestrated by
`build_options_chain()` in the public builder module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR
from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT
from volatility_trading.config.orats.ftp_schemas import STRIKES_SCHEMA_SPEC as spec

from .transforms import (
    apply_bounds_drop,
    apply_bounds_null,
    count_rows,
    count_rows_any_oob,
    dedupe_on_keys,
    fmt_int,
    log_before_after,
    log_total_missing,
)
from .io import (
    get_options_chain_path,
    scan_monies_implied_intermediate,
    scan_strikes_intermediate,
)

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """Internal mutable counters used during build (populated only if enabled)."""

    n_rows_input: int | None = None
    n_rows_after_dedupe: int | None = None

    n_rows_yield_input: int | None = None
    n_rows_yield_after_dedupe: int | None = None
    n_rows_join_missing_yield: int | None = None

    n_rows_after_trading: int | None = None
    n_rows_after_hard: int | None = None


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------

def scan_inputs(
    *,
    inter_root: Path,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None,
    collect_stats: bool,
    stats: BuildStats,
) -> pl.LazyFrame:
    lf = scan_strikes_intermediate(
        inter_root=inter_root,
        ticker=ticker,
        years=years
    )

    if collect_stats:
        lf = lf.cache()
        stats.n_rows_input = count_rows(lf)
        logger.info(
            "Input rows (strikes intermediate) ticker=%s rows=%s",
            ticker,
            fmt_int(stats.n_rows_input),
        )

    return lf


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

    logger.info(
        "Merging dividend yield from monies_implied for ticker=%s",
        ticker
    )

    lf_yield = scan_monies_implied_intermediate(
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
    lf_yield = (
        lf_yield
        .select(["ticker", "trade_date", "expiry_date", "yield_rate"])
        .rename({"yield_rate": "_dividend_yield_api"})
    )

    lf = lf.join(
        lf_yield,
        on=["ticker", "trade_date", "expiry_date"],
        how="left"
    )

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
        pl.coalesce([pl.col("_dividend_yield_api"), pl.col("dividend_yield")])
        .alias("dividend_yield")
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


def add_derived_features(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        dte=(pl.col("expiry_date") - pl.col("trade_date")).dt.total_days(),
        moneyness_ks=pl.col("strike") / pl.col("spot_price"),
        call_mid_price=(
            (pl.col("call_bid_price") + pl.col("call_ask_price")) / 2.0
        ),
        put_mid_price=(pl.col("put_bid_price") + pl.col("put_ask_price")) / 2.0,
        call_spread=pl.col("call_ask_price") - pl.col("call_bid_price"),
        put_spread=pl.col("put_ask_price") - pl.col("put_bid_price"),
    )

    lf = lf.with_columns(
        call_rel_spread=(
            pl.when(
                (pl.col("call_mid_price") > 0) & (pl.col("call_spread") >= 0)
            )
            .then(pl.col("call_spread") / pl.col("call_mid_price"))
            .otherwise(None)
        ),
        put_rel_spread=(
            pl.when(
                (pl.col("put_mid_price") > 0) & (pl.col("put_spread") >= 0)
            )
            .then(pl.col("put_spread") / pl.col("put_mid_price"))
            .otherwise(None)
        ),
    )

    return lf


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


def add_put_greeks(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Derive put Greeks via European put–call parity."""
    disc_q = (-pl.col("dividend_yield") * pl.col("yte")).exp()
    disc_r = (-pl.col("risk_free_rate") * pl.col("yte")).exp()

    return lf.with_columns(
        put_delta=pl.col("call_delta") - disc_q,
        put_gamma=pl.col("call_gamma"),
        put_vega=pl.col("call_vega"),
        put_theta=(
            pl.col("call_theta")
            + (
                pl.col("dividend_yield") * pl.col("spot_price") * disc_q
                - pl.col("risk_free_rate") * pl.col("strike") * disc_r
            )
            / CALENDAR_DAYS_PER_YEAR
        ),
        put_rho=(
            pl.col("call_rho") - pl.col("yte") * pl.col("strike") * disc_r / 100
        ),
    )


def add_put_greeks_simple(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add put Greeks using a minimal convention.

    This is useful when vendor Greeks are defined per-strike with a single
    greek set (ORATS philosophy). We keep gamma/vega/theta/rho identical
    between calls and puts and adjust delta by a constant shift.
    """
    return lf.with_columns(
        put_delta=pl.col("call_delta") - 1.0,
        put_gamma=pl.col("call_gamma"),
        put_vega=pl.col("call_vega"),
        put_theta=pl.col("call_theta"),
        put_rho=pl.col("call_rho"),
    )


def collect_and_write(
    *,
    lf: pl.LazyFrame,
    proc_root: Path,
    ticker: str,
    columns: Sequence[str],
) -> tuple[pl.DataFrame, Path]:
    lf = lf.sort(["trade_date", "expiry_date", "strike"])
    df = lf.select(list(columns)).collect()

    out_path = get_options_chain_path(proc_root=proc_root, ticker=ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing processed options chain: %s (rows=%s, cols=%s)",
        out_path,
        fmt_int(df.height),
        fmt_int(len(df.columns)),
    )

    df.write_parquet(out_path)
    return df, out_path