"""Input scan step for processed daily-features build orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ...shared.io import scan_endpoint_intermediate
from ...shared.log_fmt import fmt_int
from ...shared.stats import count_rows
from ..config import DAILY_FEATURES_ENDPOINT_COLUMNS
from ..transforms import apply_unit_multipliers

logger = logging.getLogger(__name__)


def scan_inputs(
    *,
    inter_api_root: Path,
    ticker: str,
    endpoints: Sequence[str],
    collect_stats: bool,
    stats_input_by_endpoint: dict[str, int] | None,
) -> dict[str, pl.LazyFrame]:
    """Scan and select minimal columns for each endpoint (lazy)."""
    out: dict[str, pl.LazyFrame] = {}

    for ep in endpoints:
        try:
            lf = scan_endpoint_intermediate(
                inter_api_root=inter_api_root,
                endpoint=ep,
                ticker=ticker,
            )
        except FileNotFoundError:
            logger.info("Scan: missing endpoint=%s ticker=%s", ep, ticker)
            continue

        keep = DAILY_FEATURES_ENDPOINT_COLUMNS.get(ep)
        if keep is not None:
            lf = lf.select(list(keep))

        # --- NEW: normalize units (percent -> decimal etc.) ---
        lf = apply_unit_multipliers(lf, endpoint=ep)

        if collect_stats and stats_input_by_endpoint is not None:
            lf = lf.cache()
            n = count_rows(lf)
            stats_input_by_endpoint[ep] = n
            logger.info(
                "Input rows endpoint=%s ticker=%s rows=%s",
                ep,
                ticker,
                fmt_int(n),
            )

        out[ep] = lf

    return out
