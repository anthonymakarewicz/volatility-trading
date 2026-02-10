"""Build processed ORATS options-chain panels.

This module turns **intermediate** ORATS *strikes* data (FTP) into a cleaned,
analysis-ready **processed** options-chain dataset for a single underlying.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from volatility_trading.config.instruments import OPTION_EXERCISE_STYLE
from volatility_trading.config.paths import INTER_ORATS_API

from ..shared.log_fmt import fmt_int
from ..shared.manifest import write_manifest_json
from . import steps
from .config import OPTIONS_CHAIN_CORE_COLUMNS
from .manifest import build_manifest_payload
from .types import BuildOptionsChainResult, BuildStats

logger = logging.getLogger(__name__)


def build(
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
    """Build a processed options-chain panel for one ticker.

    Args:
        inter_root: Root of intermediate ORATS strikes partitions.
        proc_root: Root directory where processed parquet is written.
        ticker: Underlying symbol (for example `"SPX"`).
        years: Optional subset of years to include.
        dte_min: Minimum allowed DTE.
        dte_max: Maximum allowed DTE.
        moneyness_min: Minimum allowed strike/spot moneyness.
        moneyness_max: Maximum allowed strike/spot moneyness.
        monies_implied_inter_root: Intermediate ORATS API root used to merge
            dividend yield from `monies_implied`.
        merge_dividend_yield: Enable/disable dividend-yield merge step.
        derive_put_greeks: Derive put greeks via parity when `True`, otherwise
            use the simplified put-greek convention.
        collect_stats: Collect optional row-count stats during build.
        columns: Optional explicit output column list.

    Returns:
        Build summary including output path, duration, row count, and optional
        stats counters.
    """
    inter_root_p = Path(inter_root)
    proc_root_p = Path(proc_root)
    monies_root_p = (
        Path(monies_implied_inter_root)
        if monies_implied_inter_root is not None
        else None
    )

    t0 = time.perf_counter()
    stats = BuildStats()

    # --- 1) Scan intermediate per-year parquet files lazily ---
    lf = steps.scan_inputs(
        inter_root=inter_root_p,
        ticker=ticker,
        years=years,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 2) Optional OPRA-root filtering (e.g. SPX vs SPXW) ---
    lf = steps.filter_preferred_opra_root(lf=lf, ticker=ticker)

    # --- 3) Remove duplicates rows & remove nulls in core keys ---
    lf = steps.dedupe_options_chain(
        lf=lf,
        ticker=ticker,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 4) Optionally replace dividend_yield using ORATS API monies_implied ---
    lf = steps.merge_dividend_yield(
        lf=lf,
        ticker=ticker,
        monies_implied_inter_root=monies_root_p,
        merge_dividend_yield=merge_dividend_yield,
        collect_stats=collect_stats,
        stats=stats,
    )

    # --- 5) Unify spot for index vs stock/ETF ---
    lf = steps.unify_spot_price(lf=lf)

    # --- 6) Bounds (NULL then DROP) ---
    lf = steps.apply_bounds(lf=lf, ticker=ticker, collect_stats=collect_stats)

    # --- 7) Derived features: DTE, moneyness, mids, spreads, rel spreads ---
    lf = steps.add_derived_features(lf=lf)

    # --- 8) Trading filters then hard sanity filters ---
    lf = steps.apply_filters(
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
        lf = steps.add_put_greeks(lf=lf)
    else:
        lf = steps.add_put_greeks_simple(lf=lf)

    # --- 10) Final column selection & materialisation ---
    cols = OPTIONS_CHAIN_CORE_COLUMNS if columns is None else tuple(columns)

    df, out_path = steps.collect_and_write(
        lf=lf,
        proc_root=proc_root_p,
        ticker=ticker,
        columns=cols,
    )

    # --- 10B) Sidecar manifest (build metadata) ---
    put_greeks_mode = "parity" if derive_put_greeks else "unified"
    exercise_style = OPTION_EXERCISE_STYLE.get(str(ticker).upper(), "AM")

    manifest_payload = build_manifest_payload(
        ticker=str(ticker),
        inter_root=inter_root_p,
        proc_root=proc_root_p,
        columns=list(df.columns),
        n_rows_written=int(df.height),
        put_greeks_mode=put_greeks_mode,
        exercise_style=exercise_style,
        merge_dividend_yield=merge_dividend_yield,
        monies_implied_inter_root=monies_root_p,
        years=years,
        dte_min=dte_min,
        dte_max=dte_max,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max,
        stats={
            "n_rows_input": stats.n_rows_input,
            "n_rows_after_dedupe": stats.n_rows_after_dedupe,
            "n_rows_yield_input": stats.n_rows_yield_input,
            "n_rows_yield_after_dedupe": stats.n_rows_yield_after_dedupe,
            "n_rows_join_missing_yield": stats.n_rows_join_missing_yield,
            "n_rows_after_trading": stats.n_rows_after_trading,
            "n_rows_after_hard": stats.n_rows_after_hard,
        },
    )

    manifest_path = write_manifest_json(
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
        "Finished building options chain ticker=%s rows_written=%s duration_s=%.2f",
        result.ticker,
        fmt_int(result.n_rows_written),
        result.duration_s,
    )

    return result
