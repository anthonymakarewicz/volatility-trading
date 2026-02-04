"""
Private IO helpers for the ORATS daily-features builder.

This module contains:
- intermediate scans (ORATS API endpoints)
- processed output path resolution
"""

from __future__ import annotations

import logging

import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Intermediate scanning
# ----------------------------------------------------------------------------

def scan_endpoint_intermediate(
    inter_api_root: Path | str,
    *,
    endpoint: str,
    ticker: str,
) -> pl.LazyFrame:
    """Lazy scan of ORATS API intermediate for one endpoint and one ticker.

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
            f"API intermediate not found for endpoint={endpoint!r} "
            f"ticker={ticker!r}: {path}"
        )

    logger.info(
        "Reading API intermediate endpoint=%s ticker=%s: %s",
        endpoint,
        ticker,
        path
    )
    return pl.scan_parquet(str(path))


# ----------------------------------------------------------------------------
# Processed output path
# ----------------------------------------------------------------------------

def get_daily_features_path(proc_root: Path, ticker: str) -> Path:
    t = str(ticker).strip()
    if not t:
        raise ValueError("ticker must be non-empty")
    return proc_root / f"underlying={t}" / "part-0000.parquet"