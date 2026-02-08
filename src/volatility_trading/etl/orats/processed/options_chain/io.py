""" ""volatility_trading.etl.orats.processed.options_chain_io

This module contains intermediate scans for FTP strikes whose behaviour requires
placing it in options_chain/ rather than shared/
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def scan_strikes_intermediate(
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
        raise FileNotFoundError(f"No ORATS directory found for {ticker!r} at {root}")

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
        raise FileNotFoundError(f"No ORATS files found for {ticker!r} under {root}")

    # allow schema evolution across years (e.g. cOpra/pOpra appear later)
    return pl.concat(scans, how="diagonal")
