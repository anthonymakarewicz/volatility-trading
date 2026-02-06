"""
Build processed ORATS daily-features panels.

This module turns **intermediate** ORATS API endpoint data into a cleaned,
analysis-ready **processed** daily-features dataset for a single underlying.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..shared.log_fmt import fmt_int

from .config import DAILY_FEATURES_CORE_COLUMNS
from .manifest import write_manifest_json
from .types import BuildDailyFeaturesResult, BuildStats
from . import steps

logger = logging.getLogger(__name__)


def build(
    *,
    inter_api_root: Path | str,
    proc_root: Path | str,
    ticker: str,
    endpoints: Sequence[str] = ("summaries", "hvs"),
    prefix_endpoint_cols: bool = True,
    priority_endpoints: Sequence[str] | None = None,
    collect_stats: bool = False,
    columns: Sequence[str] | None = DAILY_FEATURES_CORE_COLUMNS,
) -> BuildDailyFeaturesResult:
    """Build a cleaned, WIDE daily-features panel for a single ticker.

    Intermediate input layout
    -------------------------
    inter_api_root/
        endpoint=<endpoint>/underlying=<TICKER>/part-0000.parquet

    Processed output
    ----------------
    proc_root/
        underlying=<TICKER>/part-0000.parquet

    The processed panel:
    - scans intermediate endpoint panels
    - selects minimal columns per endpoint
    - de-dupes each endpoint on (ticker, trade_date)
    - builds a key spine from union of keys across endpoints
    - left-joins endpoints onto the spine
    - canonicalizes output columns (unprefixed; coalesce on collisions)
    - materializes a single wide daily-features panel
    - writes a manifest.json sidecar with build metadata
    """
    inter_api_root_p = Path(inter_api_root)
    proc_root_p = Path(proc_root)

    if not endpoints:
        raise ValueError("endpoints must be non-empty")

    if priority_endpoints is not None:
        missing = [ep for ep in priority_endpoints if ep not in endpoints]
        if missing:
            raise ValueError(
                "priority_endpoints must be a subset of endpoints; "
                f"missing={missing}"
            )

    t0 = time.perf_counter()
    stats = BuildStats()

    # --------------------------------------------------------------------- #
    # 1) Scan each endpoint into its own (ticker, trade_date, ...) panel
    # --------------------------------------------------------------------- #
    lfs = steps.scan_inputs(
        inter_api_root=inter_api_root_p,
        ticker=ticker,
        endpoints=endpoints,
        collect_stats=collect_stats,
        stats_input_by_endpoint=stats.n_rows_input_by_endpoint
        if collect_stats
        else None,
    )

    # --------------------------------------------------------------------- #
    # 2) Dedupe each endpoint on (ticker, trade_date)
    # --------------------------------------------------------------------- #
    for ep in list(lfs.keys()):
        lfs[ep] = steps.dedupe_endpoint(
            lf=lfs[ep],
            ticker=ticker,
            endpoint=ep,
            collect_stats=collect_stats,
            stats_after_dedupe_by_endpoint=stats.n_rows_after_dedupe_by_endpoint
            if collect_stats
            else None,
        )

    # --------------------------------------------------------------------- #
    # 3) Apply bounds (null/drop) per endpoint
    # --------------------------------------------------------------------- #
    for ep in list(lfs.keys()):
        lfs[ep] = steps.apply_bounds(
            lf=lfs[ep],
            ticker=ticker,
            endpoint=ep,
            collect_stats=collect_stats,
        )

    # --------------------------------------------------------------------- #
    # 4) Build key spine = union of keys across endpoints
    # --------------------------------------------------------------------- #
    stats_n_rows_spine: list[int] | None = [] if collect_stats else None

    spine = steps.build_key_spine(
        lfs=lfs,
        endpoints=endpoints,
        collect_stats=collect_stats,
        stats_n_rows_spine=stats_n_rows_spine,
    )

    if collect_stats and stats_n_rows_spine:
        stats.n_rows_spine = int(stats_n_rows_spine[-1])

    # --------------------------------------------------------------------- #
    # 5) Left-join all endpoints onto the spine (treat all equally)
    # --------------------------------------------------------------------- #
    lf = steps.join_endpoints_on_spine(
        spine=spine,
        lfs=lfs,
        ticker=ticker,
        endpoints=endpoints,
        prefix_cols=prefix_endpoint_cols,
        collect_stats=collect_stats,
        # Reuse "after dedupe" dict as join stats dict.
        stats_n_rows_endpoints=(
            stats.n_rows_after_dedupe_by_endpoint
            if collect_stats
            else None
        ),
    )

    # --------------------------------------------------------------------- #
    # 6) Canonicalize output columns (unprefixed) and write
    # --------------------------------------------------------------------- #
    output_cols = (
        DAILY_FEATURES_CORE_COLUMNS
        if columns is None
        else tuple(columns)
    )

    lf = steps.canonicalize_columns(
        lf=lf,
        endpoints=endpoints,
        output_columns=output_cols,
        prefix_cols=prefix_endpoint_cols,
        priority_endpoints=priority_endpoints,
    )

    df, out_path = steps.collect_and_write(
        lf=lf,
        proc_root=proc_root_p,
        ticker=ticker,
        columns=output_cols,
    )

    if collect_stats:
        stats.n_rows_written = int(df.height)
        stats.n_rows_input_total = int(
            sum(stats.n_rows_input_by_endpoint.values())
        )

    # --------------------------------------------------------------------- #
    # Missing endpoints reporting (manifest)
    # --------------------------------------------------------------------- #
    endpoints_used = [ep for ep in endpoints if ep in lfs]
    missing_endpoints = [ep for ep in endpoints if ep not in lfs]

    manifest_payload = {
        "schema_version": 1,
        "dataset": "orats_daily_features",
        "ticker": str(ticker),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "inter_api_root": str(inter_api_root_p),
        "proc_root": str(proc_root_p),
        "endpoints": list(endpoints),
        "endpoints_used": list(endpoints_used),
        "missing_endpoints": list(missing_endpoints),
        "prefix_endpoint_cols": bool(prefix_endpoint_cols),
        "columns": list(df.columns),
        "n_rows_written": int(df.height),
        "stats": {
            "n_rows_input_total": stats.n_rows_input_total,
            "n_rows_spine": stats.n_rows_spine,
            "n_rows_input_by_endpoint": dict(stats.n_rows_input_by_endpoint),
            "n_rows_after_dedupe_by_endpoint": dict(
                stats.n_rows_after_dedupe_by_endpoint
            ),
        }
        if collect_stats
        else None,
    }

    manifest_path = write_manifest_json(
        out_dir=out_path.parent,
        payload=manifest_payload,
    )
    logger.info("Wrote daily features manifest: %s", manifest_path)

    result = BuildDailyFeaturesResult(
        ticker=str(ticker),
        out_path=out_path,
        duration_s=time.perf_counter() - t0,
        n_rows_written=int(df.height),
        n_rows_input_total=stats.n_rows_input_total if collect_stats else None,
        n_rows_spine=stats.n_rows_spine if collect_stats else None,
        n_rows_input_by_endpoint=(
            dict(stats.n_rows_input_by_endpoint)
            if collect_stats
            else {}
        ),
        n_rows_after_dedupe_by_endpoint=(
            dict(stats.n_rows_after_dedupe_by_endpoint)
            if collect_stats
            else {}
        ),
    )

    logger.info(
        "Finished building daily features ticker=%s rows_written=%s "
        "duration_s=%.2f",
        result.ticker,
        fmt_int(result.n_rows_written),
        result.duration_s,
    )

    return result