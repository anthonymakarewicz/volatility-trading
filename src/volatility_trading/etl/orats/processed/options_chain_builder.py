"""Build processed ORATS options-chain panels.

This module turns **intermediate** ORATS *strikes* data (FTP) into a cleaned,
analysis-ready **processed** options-chain dataset for a single underlying.
"""
from __future__ import annotations

import logging
import time
import json
from datetime import datetime, timezone
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR
from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT
from volatility_trading.config.paths import INTER_ORATS_API
from volatility_trading.config.orats.ftp_schemas import (
    STRIKES_SCHEMA_SPEC as spec
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

OPTIONS_CHAIN_CORE_COLUMNS: tuple[str, ...] = (
    # identifiers / dates
    "ticker",
    "trade_date",
    "expiry_date",
    "dte",
    "yte",

    # underlying & strike
    "underlying_price",
    "spot_price",
    "strike",

    # volume & open interest
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",

    # prices
    "call_bid_price",
    "call_mid_price",
    "call_model_price",
    "call_ask_price",
    "call_rel_spread",
    "put_bid_price",
    "put_mid_price",
    "put_model_price",
    "put_ask_price",
    "put_rel_spread",

    # vols
    "smoothed_iv",
    "call_mid_iv",
    "put_mid_iv",

    # greeks (C + parity-derived P)
    "call_delta",
    "call_gamma",
    "call_theta",
    "call_vega",
    "call_rho",
    "put_delta",
    "put_gamma",
    "put_theta",
    "put_vega",
    "put_rho",

    # curves
    "risk_free_rate",
    "dividend_yield",
)


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildOptionsChainResult:
    """Summary of a processed options-chain build for one ticker.

    Notes
    -----
    When `collect_stats=False`, most counters are left as None to avoid
    additional scans.
    """
    ticker: str
    out_path: Path
    duration_s: float

    # Final materialised output
    n_rows_written: int

    # Optional stats (only when collect_stats=True)
    n_rows_input: int | None
    n_rows_after_dedupe: int | None

    n_rows_yield_input: int | None
    n_rows_yield_after_dedupe: int | None
    n_rows_join_missing_yield: int | None

    n_rows_after_trading: int | None
    n_rows_after_hard: int | None


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

@dataclass
class _BuildStats:
    """Internal mutable counters used during build (populated only if enabled)."""

    n_rows_input: int | None = None
    n_rows_after_dedupe: int | None = None

    n_rows_yield_input: int | None = None
    n_rows_yield_after_dedupe: int | None = None
    n_rows_join_missing_yield: int | None = None

    n_rows_after_trading: int | None = None
    n_rows_after_hard: int | None = None


# -- logging / stats helpers --------------------------------------------------

def _count_rows(lf: pl.LazyFrame) -> int:
    """Count rows in a LazyFrame (forces a small collect)."""
    return int(lf.select(pl.len()).collect().item())


def _fmt_int(n: int | None) -> str:
    """Format integers with thousands separators for logging."""
    if n is None:
        return "NA"
    return f"{int(n):,}"


def _log_before_after(
    *,
    label: str,
    ticker: str,
    before: int | None,
    after: int | None,
    removed_word: str = "removed",
) -> None:
    """Log a standard before/after counter line with percent removed."""
    if before is None or after is None:
        return
    removed = int(before) - int(after)
    pct = (100.0 * removed / before) if before else 0.0
    logger.info(
        "%s ticker=%s before=%s after=%s %s=%s (%.2f%%)",
        label,
        ticker,
        _fmt_int(before),
        _fmt_int(after),
        removed_word,
        _fmt_int(removed),
        pct,
    )


def _log_total_missing(
    *,
    label: str,
    ticker: str,
    total: int | None,
    missing: int | None,
    total_word: str = "rows",
    missing_word: str = "missing",
) -> None:
    """Log a standard total/missing counter line with percent missing."""
    if total is None or missing is None:
        return
    pct = (100.0 * int(missing) / int(total)) if total else 0.0
    logger.info(
        "%s ticker=%s %s=%s %s=%s (%.2f%%)",
        label,
        ticker,
        total_word,
        _fmt_int(total),
        missing_word,
        _fmt_int(missing),
        pct,
    )


# -- bounds helpers -----------------------------------------------------------

def _apply_bounds_null(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Set out-of-bounds numeric values to null (row survives).

    Notes
    -----
    - Missing columns are ignored.
    - Nulls remain null.
    """
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


def _apply_bounds_drop(
    lf: pl.LazyFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None,
) -> pl.LazyFrame:
    """Drop rows that violate structural bounds (row is removed).

    Notes
    -----
    - Missing columns are ignored.
    - For DROP bounds, nulls are treated as invalid (row is dropped).
    """
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


def _count_rows_any_oob(
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


# -- IO / scanning ------------------------------------------------------------

def _write_manifest_json(*, out_dir: Path, payload: dict) -> Path:
    """Write a manifest.json sidecar next to the processed parquet.

    The manifest captures *how* the dataset was built (key parameters and
    switches) so downstream consumers (QC, backtests) can reliably
    reproduce/interpret the output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return path


def _scan_strikes_intermediate(
    inter_root: Path | str,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None = None,
) -> pl.LazyFrame:
    """Build a lazy scan over intermediate Strikes files for a single ticker.

    Expected intermediate layout
    ----------------------------
    inter_root/
        underlying=<TICKER>/year=<YYYY>/part-0000.parquet
    """
    inter_root = Path(inter_root)
    root = inter_root / f"underlying={ticker}"

    if not root.exists():
        raise FileNotFoundError(
            f"No ORATS directory found for {ticker!r} at {root}"
        )

    logger.info("Building ORATS options chain for ticker=%s", ticker)
    logger.info("Reading intermediate from: %s", root)

    # optional year whitelist (as strings, e.g. {"2009", "2010"})
    year_whitelist = {str(y) for y in years} if years is not None else None

    scans: list[pl.LazyFrame] = []

    for year_dir in sorted(root.glob("year=*")):
        if not year_dir.is_dir():
            continue

        # year_dir.name is like "year=2009"
        year_name = year_dir.name.split("=", 1)[-1]

        if year_whitelist is not None and year_name not in year_whitelist:
            continue

        part = year_dir / "part-0000.parquet"
        if not part.exists():
            logger.debug("Skipping missing intermediate file: %s", part)
            continue

        logger.debug("Scanning intermediate file: %s", part)
        scans.append(pl.scan_parquet(str(part)))

    if not scans:
        raise FileNotFoundError(
            f"No ORATS files found for {ticker!r} under {root}"
        )

    # allow schema evolution across years (e.g. cOpra/pOpra appear later)
    return pl.concat(scans, how="diagonal")


def _scan_monies_implied_intermediate(
    inter_api_root: Path | str,
    ticker: str,
    *,
    endpoint: str = "monies_implied",
) -> pl.LazyFrame:
    """Lazy scan of ORATS API intermediate for monies_implied for one ticker.

    Expected intermediate layout
    ----------------------------
    inter_api_root/
        endpoint=<endpoint>/underlying=<TICKER>/part-0000.parquet
    """
    inter_api_root = Path(inter_api_root)
    path = (
        inter_api_root 
        / f"endpoint={endpoint}" 
        / f"underlying={ticker}" 
        / "part-0000.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"monies_implied intermediate not found for {ticker!r}: {path}"
        )

    logger.info(
        "Reading monies_implied intermediate for ticker=%s: %s", 
        ticker, 
        path
    )
    return pl.scan_parquet(str(path))


def _get_options_chain_path(proc_root: Path, ticker: str) -> Path:
    t = str(ticker).strip()
    if not t:
        raise ValueError("ticker must be non-empty")
    return proc_root / f"underlying={t}" / "part-0000.parquet"


# -- generic transforms -------------------------------------------------------

def _dedupe_on_keys(
    lf: pl.LazyFrame,
    *,
    key_common: Sequence[str],
    key_when_opra_present: Sequence[str] | None = None,
    opra_nonnull_cols: Sequence[str] | None = None,
    stable_sort: bool = False,
) -> pl.LazyFrame:
    """Drop exact duplicates using best-available key columns.

    Notes
    -----
    This is a *mechanical* de-duplication helper for processed-building:
    - Rows with nulls in `key_common` are dropped.
    - If OPRA codes are present and you provide `key_when_opra_present` and
      `opra_nonnull_cols`, then:
        * rows where all OPRA cols are non-null are de-duped on
          `key_when_opra_present`
        * rows missing OPRA cols are de-duped on `key_common`

    If you need "latest wins" semantics, you must include a timestamp column
    and sort by it before calling `.unique(keep="last")`.
    """
    schema = lf.collect_schema()
    cols = set(schema.names())

    key_common_eff = [c for c in key_common if c in cols]
    if not key_common_eff:
        raise ValueError(
            f"_dedupe_on_keys: none of key_common "
            f"columns exist: {list(key_common)}"
        )

    # Drop rows with nulls in the always-required keys.
    lf = lf.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in key_common_eff])
    )

    def _unique_on(subset: Sequence[str], lf_in: pl.LazyFrame) -> pl.LazyFrame:
        subset_eff = [c for c in subset if c in cols]
        if not subset_eff:
            return lf_in
        # Optional deterministic ordering (can be expensive on large scans).
        if stable_sort:
            lf_in = lf_in.sort(subset_eff)
        return lf_in.unique(subset=subset_eff, maintain_order=True)

    # No OPRA-aware logic requested.
    if not (key_when_opra_present and opra_nonnull_cols):
        return _unique_on(key_common_eff, lf)

    opra_cols_eff = [c for c in opra_nonnull_cols if c in cols]
    if len(opra_cols_eff) != len(opra_nonnull_cols):
        # OPRA columns not available in this scan; fall back to common key.
        return _unique_on(key_common_eff, lf)

    has_opra_expr = pl.all_horizontal(
        [pl.col(c).is_not_null() for c in opra_cols_eff]
    )

    lf_with = _unique_on(key_when_opra_present, lf.filter(has_opra_expr))
    lf_without = _unique_on(key_common_eff, lf.filter(~has_opra_expr))

    return pl.concat([lf_with, lf_without], how="vertical")


# -- pipeline steps ----------------------------------------------------------

def _step_scan_inputs(
    *,
    inter_root: Path,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None,
    collect_stats: bool,
    stats: _BuildStats,
) -> pl.LazyFrame:
    lf = _scan_strikes_intermediate(
        inter_root=inter_root, 
        ticker=ticker,
        years=years
    )

    if collect_stats:
        lf = lf.cache()
        stats.n_rows_input = _count_rows(lf)
        logger.info(
            "Input rows (strikes intermediate) ticker=%s rows=%s",
            ticker,
            _fmt_int(stats.n_rows_input),
        )

    return lf


def _step_filter_preferred_opra_root(
    *, 
    lf: pl.LazyFrame, 
    ticker: str
) -> pl.LazyFrame:
    preferred_root = PREFERRED_OPRA_ROOT.get(ticker)
    if preferred_root is None:
        return lf

    return lf.filter(
        # For older years, call_opra is null / absent; keep those rows.
        pl.col("call_opra").is_null()
        | pl.col("call_opra").str.starts_with(preferred_root)
    )


def _step_dedupe_options_chain(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    collect_stats: bool,
    stats: _BuildStats,
) -> pl.LazyFrame:
    logger.info(
        "Applying key null-checks and de-duplication for ticker=%s", 
        ticker
    )

    n_before: int | None = _count_rows(lf) if collect_stats else None

    lf = _dedupe_on_keys(
        lf,
        key_common=["ticker", "trade_date", "expiry_date", "strike"],
        key_when_opra_present=[
            "ticker", "trade_date", "strike", "call_opra", "put_opra"
        ],
        opra_nonnull_cols=["call_opra", "put_opra"],
        stable_sort=False,
    )

    if collect_stats:
        stats.n_rows_after_dedupe = _count_rows(lf)
        _log_before_after(
            label="Dedupe (options chain)",
            ticker=ticker,
            before=n_before,
            after=stats.n_rows_after_dedupe,
            removed_word="removed",
        )

    return lf


def _step_merge_dividend_yield(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    monies_implied_inter_root: Path | None,
    merge_dividend_yield: bool,
    collect_stats: bool,
    stats: _BuildStats,
) -> pl.LazyFrame:
    if not merge_dividend_yield or monies_implied_inter_root is None:
        return lf

    logger.info(
        "Merging dividend yield from monies_implied for ticker=%s", 
        ticker
    )

    lf_yield = _scan_monies_implied_intermediate(
        inter_api_root=monies_implied_inter_root,
        ticker=ticker,
        endpoint="monies_implied",
    )

    if collect_stats:
        lf_yield = lf_yield.cache()
        stats.n_rows_yield_input = _count_rows(lf_yield)
        logger.info(
            "Input rows (monies_implied intermediate) ticker=%s rows=%s",
            ticker,
            _fmt_int(stats.n_rows_yield_input),
        )

    lf_yield = _dedupe_on_keys(
        lf_yield,
        key_common=["ticker", "trade_date", "expiry_date"],
        stable_sort=False,
    )

    if collect_stats:
        stats.n_rows_yield_after_dedupe = _count_rows(lf_yield)
        _log_before_after(
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
        .rename(
            {"yield_rate": "_dividend_yield_api"}
        )
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
            _log_total_missing(
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
        .alias(
            "dividend_yield"
        )
    ).drop(["_dividend_yield_api"])

    return lf


def _step_unify_spot_price(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Unify spot_price across index vs stock/ETF conventions."""
    return lf.with_columns(
        spot_price=pl.when(
            pl.col("spot_price").is_not_null() & (pl.col("spot_price") > 0)
        )
        .then(pl.col("spot_price"))
        .otherwise(pl.col("underlying_price"))
    )


def _step_apply_bounds(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    collect_stats: bool,
) -> pl.LazyFrame:
    # 6A) Bounds NA replacement (NULL)
    bounds_null = getattr(spec, "bounds_null_canonical", None)

    n_total: int | None = None
    if collect_stats:
        n_total, n_rows_oob = _count_rows_any_oob(lf, bounds=bounds_null)
        if n_total is not None and n_rows_oob is not None:
            _log_total_missing(
                label="Bounds null",
                ticker=ticker,
                total=n_total,
                missing=n_rows_oob,
                total_word="rows",
                missing_word="rows_oob",
            )

    lf = _apply_bounds_null(lf, bounds=bounds_null)

    # 6B) Bounds filters (DROP)
    bounds_drop = getattr(spec, "bounds_drop_canonical", None)
    # Reuse the total row count from bounds-null stats when available.
    n_before_drop: int | None
    if collect_stats:
        n_before_drop = n_total if n_total is not None else _count_rows(lf)
    else:
        n_before_drop = None

    lf = _apply_bounds_drop(lf, bounds=bounds_drop)

    if collect_stats:
        n_after_drop = _count_rows(lf)
        _log_before_after(
            label="Bounds drop",
            ticker=ticker,
            before=n_before_drop,
            after=n_after_drop,
            removed_word="dropped",
        )

    return lf


def _step_add_derived_features(
    *,
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
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


def _step_apply_filters(
    *,
    lf: pl.LazyFrame,
    ticker: str,
    dte_min: int,
    dte_max: int,
    moneyness_min: float,
    moneyness_max: float,
    collect_stats: bool,
    stats: _BuildStats,
) -> pl.LazyFrame:
    # 8A) Trading band filters
    n_before_trading: int | None = _count_rows(lf) if collect_stats else None

    lf = lf.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),
    )

    if collect_stats:
        stats.n_rows_after_trading = _count_rows(lf)
        _log_before_after(
            label="Trading filters",
            ticker=ticker,
            before=n_before_trading,
            after=stats.n_rows_after_trading,
            removed_word="dropped",
        )

    # 8B) Hard sanity filters
    n_before_hard: int | None = _count_rows(lf) if collect_stats else None

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
        stats.n_rows_after_hard = _count_rows(lf)
        _log_before_after(
            label="Hard sanity",
            ticker=ticker,
            before=n_before_hard,
            after=stats.n_rows_after_hard,
            removed_word="dropped",
        )

    return lf


def _step_add_put_greeks(*, lf: pl.LazyFrame) -> pl.LazyFrame:
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


def _step_add_put_greeks_simple(*, lf: pl.LazyFrame) -> pl.LazyFrame:
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


def _step_collect_and_write(
    *,
    lf: pl.LazyFrame,
    proc_root: Path,
    ticker: str,
    columns: Sequence[str] | None,
) -> tuple[pl.DataFrame, Path]:
    cols = OPTIONS_CHAIN_CORE_COLUMNS if columns is None else tuple(columns)

    lf = lf.sort(["trade_date", "expiry_date", "strike"])
    df = lf.select(cols).collect()

    out_path = _get_options_chain_path(proc_root=proc_root, ticker=ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing processed options chain: %s (rows=%s, cols=%s)",
        out_path,
        _fmt_int(df.height),
        _fmt_int(len(df.columns)),
    )

    df.write_parquet(out_path)
    return df, out_path


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_options_chain(
    *,
    inter_root: Path | str,
    proc_root: Path | str,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None = None,
    dte_min: int = 7,
    dte_max: int = 60,
    moneyness_min: float = 0.5,
    moneyness_max: float = 1.5,
    monies_implied_inter_root: Path | str | None = INTER_ORATS_API,
    merge_dividend_yield: bool = True,
    derive_put_greeks: bool = True,
    collect_stats: bool = False,
    columns: Sequence[str] | None = OPTIONS_CHAIN_CORE_COLUMNS,
) -> BuildOptionsChainResult:
    """Build a cleaned, WIDE ORATS panel for a single ticker.

    Raw (intermediate) layout
    -------------------------
    inter_root/
        underlying=<TICKER>/year=<YYYY>/part-0000.parquet

    Processed output
    ----------------
    proc_root/
        underlying=<TICKER>/part-0000.parquet

    The processed panel:
    - parses dates (trade_date, expiry_date)
    - computes DTE and K/S moneyness
    - normalises vendor names (stkPx -> spot_price, cBidPx -> call_bid_price)
    - optionally merges dividend yield from ORATS API monies_implied
      (replacing empty/invalid dividend_yield in the FTP strikes)
    - computes call/put mid prices, spreads, relative spreads
    - applies basic sanity filters
    - trims extreme DTE and moneyness
    - drops completely dead contracts (no quotes, no theo value)
    - exposes call Greeks as call_delta, call_gamma, ...
    - adds put Greeks via European put–call parity (put_delta, put_gamma)

    Parameters
    ----------
    inter_root:
        Root of intermediate ORATS-by-ticker parquet structure.
    proc_root:
        Root for processed ORATS panels.
    ticker:
        Underlying symbol (e.g. "SPX").
    years:
        Optional subset of years to include (ints or strings).
    dte_min, dte_max:
        DTE band to keep in the processed panel.
    moneyness_min, moneyness_max:
        Coarse K/S moneyness band to keep.
    monies_implied_inter_root:
        Root of intermediate ORATS API parquet output (the folder that contains
        `endpoint=monies_implied/underlying=<TICKER>/part-0000.parquet`).
        If None, the merge step is skipped.
    merge_dividend_yield:
        If True (default), replace/overwrite `dividend_yield` in the 
        options chain using `yield_rate` from the monies_implied endpoint 
        joined on (ticker, trade_date, expiry_date).
    derive_put_greeks:
        If True (default), derive put Greeks via European put–call parity using
        dividend_yield and risk_free_rate. If False, use a minimal convention:
        put_delta = call_delta - 1 and copy theta/rho/gamma/vega from the call.
    collect_stats:
        If True, collect all the rows counts from filtering operations e.g.
        removed duplicates, remove negative prices ...
    columns:
        Optional explicit list of columns for the output schema.
        If None, uses OPTIONS_CHAIN_CORE_COLUMNS.

    Returns
    -------
    BuildOptionsChainResult
        Summary including optional row-drop stats when `collect_stats=True`.
    """
    inter_root_p = Path(inter_root)
    proc_root_p = Path(proc_root)
    monies_root_p = (
        Path(monies_implied_inter_root) 
        if monies_implied_inter_root is not None 
        else None
    )

    t0 = time.perf_counter()
    stats = _BuildStats()

    # --- 1) Scan intermediate per-year parquet files lazily ---
    lf = _step_scan_inputs(
        inter_root=inter_root_p,
        ticker=ticker,
        years=years,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 2) Optional OPRA-root filtering (e.g. SPX vs SPXW) ---
    lf = _step_filter_preferred_opra_root(lf=lf, ticker=ticker)

    # --- 3) Remove duplicates rows & remove nulls in core keys ---
    lf = _step_dedupe_options_chain(
        lf=lf, 
        ticker=ticker, 
        collect_stats=collect_stats, stats=stats
    )

    # --- 4) Optionally replace dividend_yield using ORATS API monies_implied ---
    lf = _step_merge_dividend_yield(
        lf=lf,
        ticker=ticker,
        monies_implied_inter_root=monies_root_p,
        merge_dividend_yield=merge_dividend_yield,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 5) Unify spot for index vs stock/ETF ---
    lf = _step_unify_spot_price(lf=lf)

    # --- 6) Bounds (NULL then DROP) ---
    lf = _step_apply_bounds(lf=lf, ticker=ticker, collect_stats=collect_stats)

    # --- 7) Derived features: DTE, moneyness, mids, spreads, rel spreads ---
    lf = _step_add_derived_features(lf=lf)

    # --- 8) Trading filters then hard sanity filters ---
    lf = _step_apply_filters(
        lf=lf,
        ticker=ticker,
        dte_min=dte_min,
        dte_max=dte_max,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 9) Put greeks (parity-derived or minimal convention) ---
    if derive_put_greeks:
        lf = _step_add_put_greeks(lf=lf)
    else:
        lf = _step_add_put_greeks_simple(lf=lf)

    # --- 10) Final column selection & materialisation ---
    df, out_path = _step_collect_and_write(
        lf=lf,
        proc_root=proc_root_p,
        ticker=ticker,
        columns=columns,
    )

    # --- 10B) Sidecar manifest (build metadata) ---
    put_greeks_mode = "parity" if derive_put_greeks else "unified"

    manifest_payload = {
        "schema_version": 1,
        "dataset": "orats_options_chain",
        "ticker": str(ticker),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "put_greeks_mode": put_greeks_mode,
        "merge_dividend_yield": bool(merge_dividend_yield),
        "monies_implied_inter_root": str(monies_root_p) if monies_root_p else None,
        "inter_root": str(inter_root_p),
        "proc_root": str(proc_root_p),
        "years": [str(y) for y in years] if years is not None else None,
        "dte_min": int(dte_min),
        "dte_max": int(dte_max),
        "moneyness_min": float(moneyness_min),
        "moneyness_max": float(moneyness_max),
        "columns": list(df.columns),
        "n_rows_written": int(df.height),
        "stats": {
            "n_rows_input": stats.n_rows_input,
            "n_rows_after_dedupe": stats.n_rows_after_dedupe,
            "n_rows_yield_input": stats.n_rows_yield_input,
            "n_rows_yield_after_dedupe": stats.n_rows_yield_after_dedupe,
            "n_rows_join_missing_yield": stats.n_rows_join_missing_yield,
            "n_rows_after_trading": stats.n_rows_after_trading,
            "n_rows_after_hard": stats.n_rows_after_hard,
        },
    }

    manifest_path = _write_manifest_json(
        out_dir=out_path.parent,
        payload=manifest_payload,
    )

    logger.info(
        "Wrote options chain manifest: %s",
        manifest_path,
    )

    result = BuildOptionsChainResult(
        ticker=str(ticker),
        out_path=out_path,
        duration_s=time.perf_counter() - t0,
        n_rows_written=int(df.height),
        n_rows_input=stats.n_rows_input,
        n_rows_after_dedupe=stats.n_rows_after_dedupe,
        n_rows_yield_input=stats.n_rows_yield_input,
        n_rows_yield_after_dedupe=stats.n_rows_yield_after_dedupe,
        n_rows_join_missing_yield=stats.n_rows_join_missing_yield,
        n_rows_after_trading=stats.n_rows_after_trading,
        n_rows_after_hard=stats.n_rows_after_hard,
    )

    logger.info(
        "Finished building options chain ticker=%s "
        "rows_written=%s duration_s=%.2f",
        result.ticker,
        _fmt_int(result.n_rows_written),
        result.duration_s,
    )

    return result