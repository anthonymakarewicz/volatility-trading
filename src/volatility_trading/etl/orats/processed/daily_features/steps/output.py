from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ...shared.io import processed_underlying_part_path
from ...shared.log_fmt import fmt_int


logger = logging.getLogger(__name__)


def collect_and_write(
    *,
    lf: pl.LazyFrame,
    proc_root: Path,
    ticker: str,
    columns: Sequence[str],
) -> tuple[pl.DataFrame, Path]:
    lf = lf.sort(["trade_date"])
    df = lf.select(list(columns)).collect()

    out_path = processed_underlying_part_path(proc_root=proc_root, ticker=ticker)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Writing processed daily features: %s (rows=%s, cols=%s)",
        out_path,
        fmt_int(df.height),
        fmt_int(len(df.columns)),
    )

    df.write_parquet(out_path)
    return df, out_path