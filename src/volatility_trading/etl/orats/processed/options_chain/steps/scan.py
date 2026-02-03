# volatility_trading/etl/orats/processed/options_chain/_steps/scan.py

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from ..types import BuildStats
from ..io import scan_strikes_intermediate
from ..transforms import (
    count_rows,
    fmt_int,
)

logger = logging.getLogger(__name__)


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