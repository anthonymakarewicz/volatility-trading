from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

from ...shared.log_fmt import fmt_int
from ...shared.stats import count_rows

logger = logging.getLogger(__name__)


def _prefix_endpoint_cols(*, lf: pl.LazyFrame, endpoint: str) -> pl.LazyFrame:
    """Prefix non-key cols with '<endpoint>__' to avoid collisions."""
    keys = {"ticker", "trade_date"}
    schema = lf.collect_schema()
    cols = schema.names()

    renames: dict[str, str] = {}
    for c in cols:
        if c in keys:
            continue
        renames[c] = f"{endpoint}__{c}"

    return lf.rename(renames) if renames else lf


def build_key_spine(
    *,
    lfs: dict[str, pl.LazyFrame],
    endpoints: Sequence[str],
    collect_stats: bool,
    stats_n_rows_spine: list[int] | None,
) -> pl.LazyFrame:
    """Build a unique (ticker, trade_date) spine from all endpoints."""
    scans: list[pl.LazyFrame] = []

    for ep in endpoints:
        lf = lfs.get(ep)
        if lf is None:
            continue
        scans.append(lf.select(["ticker", "trade_date"]))

    if not scans:
        raise ValueError("No endpoint panels available to build key spine")

    spine = pl.concat(scans, how="vertical").unique(
        subset=["ticker", "trade_date"],
        maintain_order=True,
    )

    if collect_stats and stats_n_rows_spine is not None:
        spine = spine.cache()
        n_spine = count_rows(spine)
        stats_n_rows_spine.append(n_spine)
        logger.info("Key spine rows=%s", fmt_int(n_spine))

    return spine


def join_endpoints_on_spine(
    *,
    spine: pl.LazyFrame,
    lfs: dict[str, pl.LazyFrame],
    ticker: str,
    endpoints: Sequence[str],
    prefix_cols: bool,
    collect_stats: bool,
    stats_n_rows_endpoints: dict[str, int] | None,
) -> pl.LazyFrame:
    """Left-join all endpoints onto spine on (ticker, trade_date)."""
    lf = spine

    for ep in endpoints:
        lf_ep = lfs.get(ep)
        if lf_ep is None:
            logger.info("Join: skipping missing endpoint=%s ticker=%s", ep, ticker)
            continue

        if collect_stats and stats_n_rows_endpoints is not None:
            lf_ep = lf_ep.cache()
            n_ep = count_rows(lf_ep)
            stats_n_rows_endpoints[ep] = n_ep
            logger.info(
                "Join: endpoint=%s ticker=%s rows=%s",
                ep,
                ticker,
                fmt_int(n_ep),
            )

        if prefix_cols:
            lf_ep = _prefix_endpoint_cols(lf=lf_ep, endpoint=ep)

        logger.info("Join: merging endpoint=%s ticker=%s how=left", ep, ticker)
        lf = lf.join(
            lf_ep,
            on=["ticker", "trade_date"],
            how="left",
        )

    return lf
