"""Build processed ORATS options-chain panels.

This module turns **intermediate** ORATS *strikes* data (FTP) into a cleaned,
analysis-ready **processed** options-chain dataset for a single underlying.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from collections.abc import Iterable, Sequence
from pathlib import Path

from volatility_trading.config.paths import INTER_ORATS_API
from .config import OPTIONS_CHAIN_CORE_COLUMNS
from .io import _write_manifest_json
from .types import (
    BuildOptionsChainResult,
    _BuildStats,
)
from .steps import (
    _step_add_derived_features,
    _step_add_put_greeks,
    _step_add_put_greeks_simple,
    _step_apply_bounds,
    _step_apply_filters,
    _step_collect_and_write,
    _step_dedupe_options_chain,
    _step_filter_preferred_opra_root,
    _step_merge_dividend_yield,
    _step_scan_inputs,
    _step_unify_spot_price,
)
from .transforms import _fmt_int

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

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
        collect_stats=collect_stats,
        stats=stats,
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
    cols = OPTIONS_CHAIN_CORE_COLUMNS if columns is None else tuple(columns)

    df, out_path = _step_collect_and_write(
        lf=lf,
        proc_root=proc_root_p,
        ticker=ticker,
        columns=cols,
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