from __future__ import annotations

import logging
from collections.abc import Sequence, Iterable
from pathlib import Path

import time
from dataclasses import dataclass

import polars as pl

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR
from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT
from volatility_trading.config.orats.ftp_schemas import (
    STRIKES_SCHEMA_SPEC as spec
)

from volatility_trading.config.paths import INTER_ORATS_API

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Result object
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

    n_rows_after_trading_filter: int | None
    n_rows_after_hard_filters: int | None


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

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


def _scan_strikes_intermediate(
    inter_root: Path | str,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None = None,
) -> pl.LazyFrame:
    """
    Build a lazy scan over intermediate Strikes files for a single ticker.

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
    if years is not None:
        year_whitelist = {str(y) for y in years}
    else:
        year_whitelist = None

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


def _get_options_chain_path(proc_root: Path,ticker: str) -> Path:
    t = str(ticker).strip()
    if not t:
        raise ValueError("ticker must be non-empty")
    return proc_root / f"underlying={t}" / "part-0000.parquet"


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
            f"_dedupe_on_keys: none of key_common columns exist: {list(key_common)}"
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
    collect_stats: bool = False,
    columns: Sequence[str] | None = None,
) -> BuildOptionsChainResult:
    """
    Build a cleaned, WIDE ORATS panel for a single ticker.

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
    - normalises vendor names (stkPx -> spot_price, cBidPx -> call_bid_price, ...)
    - optionally merges dividend yield from ORATS API monies_implied 
    (replacing empty/invalid dividend_yield in the FTP strikes)
    - computes call/put mid prices, spreads, relative spreads
    - applies basic sanity filters
    - trims extreme DTE and moneyness
    - drops completely dead contracts (no quotes, no theo value)
    - exposes call Greeks as call_delta, call_gamma, ...
    - adds put Greeks via European put–call parity (put_delta, put_gamma, ...)

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
        If True (default), replace/overwrite `dividend_yield` in the options chain
        using `yield_rate` from the monies_implied endpoint joined on
        (ticker, trade_date, expiry_date).
    collect_stats:
        If True, collect all the rows counts from filtering operations e.g.
        removed duplicates, remove negative prices ...
    columns:
        Optional explicit list of columns for the output schema.
        If None, uses CORE_ORATS_WIDE_COLUMNS.
        REQUIRED_ORATS_WIDE_COLUMNS must always be included.

    Returns
    -------
    BuildOptionsChainResult
        Summary including optional row-drop stats when `collect_stats=True`.
    """
    inter_root = Path(inter_root)
    proc_root = Path(proc_root)

    t0 = time.perf_counter()

    # Stats (only filled when collect_stats=True)
    n_rows_input: int | None = None
    n_rows_after_dedupe: int | None = None

    n_rows_yield_input: int | None = None
    n_rows_yield_after_dedupe: int | None = None
    n_rows_join_missing_yield: int | None = None

    n_rows_after_trading_filter: int | None = None
    n_rows_after_hard_filters: int | None = None

    # --- 1) Scan intermediate per-year parquet files lazily ---
    lf = _scan_strikes_intermediate(
        inter_root=inter_root,
        ticker=ticker,
        years=years,
    )

    if collect_stats:
        lf = lf.cache()
        n_rows_input = _count_rows(lf)
        logger.info(
            "Input rows (strikes intermediate) ticker=%s rows=%s",
            ticker,
            _fmt_int(n_rows_input),
        )

    # --- 2) Optional OPRA-root filtering (e.g. SPX vs SPXW) ---
    preferred_root = PREFERRED_OPRA_ROOT.get(ticker)
    if preferred_root is not None:
        lf = lf.filter(
            # For older years, call_opra is null / absent; keep those rows.
            pl.col("call_opra").is_null()
            | pl.col("call_opra").str.starts_with(preferred_root)
        )

    # --- 3) Remove duplicates rows & remove Nan in core cols ---
    logger.info(
        "Applying key null-checks and de-duplication for ticker=%s",
        ticker,
    )

    n_before_dedupe: int | None = None
    if collect_stats:
        n_before_dedupe = _count_rows(lf)

    lf = _dedupe_on_keys(
        lf,
        key_common=["ticker", "trade_date", "expiry_date", "strike"],
        key_when_opra_present=[
            "ticker",
            "trade_date",
            "strike",
            "call_opra",
            "put_opra",
        ],
        opra_nonnull_cols=["call_opra", "put_opra"],
        stable_sort=False,
    )

    if collect_stats:
        n_rows_after_dedupe = _count_rows(lf)
        _log_before_after(
            label="Dedupe (options chain)",
            ticker=ticker,
            before=n_before_dedupe,
            after=n_rows_after_dedupe,
            removed_word="removed",
        )

    # --- 4) Optionally replace dividend_yield using ORATS API monies_implied ---
    if merge_dividend_yield and monies_implied_inter_root is not None:
        logger.info(
            "Merging dividend yield from monies_implied for ticker=%s",
            ticker,
        )

        lf_yield = _scan_monies_implied_intermediate(
            inter_api_root=monies_implied_inter_root,
            ticker=ticker,
            endpoint="monies_implied",
        )

        if collect_stats:
            lf_yield = lf_yield.cache()
            n_rows_yield_input = _count_rows(lf_yield)
            logger.info(
                "Input rows (monies_implied intermediate) ticker=%s rows=%s",
                ticker,
                _fmt_int(n_rows_yield_input),
            )

        n_before_yield_dedupe: int | None = None
        if collect_stats:
            n_before_yield_dedupe = _count_rows(lf_yield)

        lf_yield = _dedupe_on_keys(
            lf_yield,
            key_common=["ticker", "trade_date", "expiry_date"],
            stable_sort=False,
        )

        if collect_stats:
            n_rows_yield_after_dedupe = _count_rows(lf_yield)
            _log_before_after(
                label="Dedupe (monies_implied)",
                ticker=ticker,
                before=n_before_yield_dedupe,
                after=n_rows_yield_after_dedupe,
                removed_word="removed",
            )

        # monies_implied is expiry-specific, so join on 
        # (ticker, trade_date, expiry_date)
        lf_yield = (
            lf_yield
            .select(["ticker", "trade_date", "expiry_date", "yield_rate"])
            .rename({"yield_rate": "_dividend_yield_api"})
        )

        lf = lf.join(
            lf_yield,
            on=["ticker", "trade_date", "expiry_date"],
            how="left",
        )

        if collect_stats:
            try:
                n_total_join = _count_rows(lf)
                n_miss = _count_rows(
                    lf.filter(pl.col("_dividend_yield_api").is_null())
                )
                n_rows_join_missing_yield = n_miss
                _log_total_missing(
                    label="Yield join",
                    ticker=ticker,
                    total=n_total_join,
                    missing=n_miss,
                    total_word="rows",
                    missing_word="missing",
                )
            except Exception:
                # Best-effort stats only
                logger.debug(
                    "Yield-join stats failed for ticker=%s",
                    ticker,
                    exc_info=True
                )

        # Replace existing dividend_yield
        lf = lf.with_columns(
            pl.coalesce([
                pl.col("_dividend_yield_api"),
                pl.col("dividend_yield"),
            ]).alias("dividend_yield")
        )
        lf = lf.drop(["_dividend_yield_api"])

    # --- 5) Unify spot for index vs stock/ETF ---
    """
    For index options:
        spot_price = cash index
        underlying_price = per-expiry parity implied forward
     For stock/ETF options:
       spot_price is null or 0.0, and underlying_price is the tradable spot.
    """
    lf = lf.with_columns(
        spot_price = pl.when(
            pl.col("spot_price").is_not_null() & (pl.col("spot_price") > 0)
        )
        .then(pl.col("spot_price"))
        .otherwise(pl.col("underlying_price"))
    )

    # --- 6) Derived features: DTE, moneyness, mids, spreads, rel spreads ---
    lf = lf.with_columns(
        dte = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days(),
        moneyness_ks = pl.col("strike") / pl.col("spot_price"),
        call_mid_price = (
            (pl.col("call_bid_price") + pl.col("call_ask_price")) / 2.0
        ),
        put_mid_price = (
            (pl.col("put_bid_price") + pl.col("put_ask_price")) / 2.0
        ),
        call_spread = pl.col("call_ask_price") - pl.col("call_bid_price"),
        put_spread  = pl.col("put_ask_price") - pl.col("put_bid_price"),
    )

    lf = lf.with_columns(
        call_rel_spread = (
            pl.when((pl.col("call_mid_price") > 0) & (pl.col("call_spread") >= 0))
            .then(pl.col("call_spread") / pl.col("call_mid_price"))
            .otherwise(None)
        ),
        put_rel_spread = (
            pl.when((pl.col("put_mid_price") > 0) & (pl.col("put_spread") >= 0))
            .then(pl.col("put_spread") / pl.col("put_mid_price"))
            .otherwise(None)
        ),
    )
    
    # --- 7) Filters (two-stage) ---
    if collect_stats:
        n_before_filters = _count_rows(lf)
    else:
        n_before_filters = None

    # 7A) Trading band filters
    lf = lf.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),
    )

    if collect_stats:
        n_rows_after_trading_filter = _count_rows(lf)
        _log_before_after(
            label="Trading filters",
            ticker=ticker,
            before=n_before_filters,
            after=n_rows_after_trading_filter,
            removed_word="dropped",
        )

    # 7B) Hard sanity filters
    if collect_stats:
        n_before_hard = _count_rows(lf)
    else:
        n_before_hard = None

    lf = lf.filter(
        pl.col("spot_price") > 0,
        pl.col("strike") > 0,
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
        n_rows_after_hard_filters = _count_rows(lf)
        _log_before_after(
            label="Hard filters",
            ticker=ticker,
            before=n_before_hard,
            after=n_rows_after_hard_filters,
            removed_word="dropped",
        )

    # --- 8) Put greeks via European put–call parity ---
    disc_q = (-pl.col("dividend_yield") * pl.col("yte")).exp()
    disc_r = (-pl.col("risk_free_rate") * pl.col("yte")).exp()

    lf = lf.with_columns(
        put_delta = pl.col("call_delta") - disc_q,
        put_gamma = pl.col("call_gamma"),
        put_vega = pl.col("call_vega"),
        put_theta = (
            pl.col("call_theta") + (
                pl.col("dividend_yield") * pl.col("spot_price") * disc_q - 
                pl.col("risk_free_rate") * pl.col("strike") * disc_r
            ) / CALENDAR_DAYS_PER_YEAR
        ),
        put_rho = (
            pl.col("call_rho") - pl.col("yte") * pl.col("strike") * disc_r / 100
        )
    )

    # --- 9) Final column selection & materialisation ---
    if columns is None:
        cols = spec.keep_canonical
    else:
        cols = columns

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

    result = BuildOptionsChainResult(
        ticker=str(ticker),
        out_path=out_path,
        duration_s=time.perf_counter() - t0,
        n_rows_written=int(df.height),
        n_rows_input=n_rows_input,
        n_rows_after_dedupe=n_rows_after_dedupe,
        n_rows_yield_input=n_rows_yield_input,
        n_rows_yield_after_dedupe=n_rows_yield_after_dedupe,
        n_rows_join_missing_yield=n_rows_join_missing_yield,
        n_rows_after_trading_filter=n_rows_after_trading_filter,
        n_rows_after_hard_filters=n_rows_after_hard_filters,
    )

    logger.info(
        "Finished building options chain ticker=%s "
        "rows_written=%s duration_s=%.2f",
        result.ticker,
        _fmt_int(result.n_rows_written),
        result.duration_s,
    )

    return result