"""Input scan step for processed options-chain build orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from ...shared.log_fmt import fmt_int
from ...shared.stats import count_rows
from ..io import scan_strikes_intermediate
from ..types import BuildStats

logger = logging.getLogger(__name__)


def scan_inputs(
    *,
    inter_root: Path,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None,
    collect_stats: bool,
    stats: BuildStats,
) -> pl.LazyFrame:
    """Scan intermediate inputs and optionally collect row-count stats."""
    lf = scan_strikes_intermediate(inter_root=inter_root, ticker=ticker, years=years)

    if collect_stats:
        lf = lf.cache()
        stats.n_rows_input = count_rows(lf)
        logger.info(
            "Input rows (strikes intermediate) ticker=%s rows=%s",
            ticker,
            fmt_int(stats.n_rows_input),
        )

    return lf
