"""""volatility_trading.etl.orats.processed.options_chain_io

Private IO helpers for the ORATS options-chain builder.

This module contains:
- intermediate scans (FTP strikes, API monies_implied)
- processed output path resolution
- writing manifest sidecars
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def write_manifest_json(*, out_dir: Path, payload: dict) -> Path:
    """Write a manifest.json sidecar next to the processed parquet.

    The manifest captures *how* the dataset was built (key parameters and
    switches) so downstream consumers (QC, backtests) can reliably
    reproduce/interpret the output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return path


# ----------------------------------------------------------------------------
# Intermediate scanning
# ----------------------------------------------------------------------------

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
        raise FileNotFoundError(
            f"No ORATS directory found for {ticker!r} at {root}"
        )

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
        raise FileNotFoundError(
            f"No ORATS files found for {ticker!r} under {root}"
        )

    # allow schema evolution across years (e.g. cOpra/pOpra appear later)
    return pl.concat(scans, how="diagonal")


def scan_monies_implied_intermediate(
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


# ----------------------------------------------------------------------------
# Processed output path
# ----------------------------------------------------------------------------

def get_options_chain_path(proc_root: Path, ticker: str) -> Path:
    t = str(ticker).strip()
    if not t:
        raise ValueError("ticker must be non-empty")
    return proc_root / f"underlying={t}" / "part-0000.parquet"