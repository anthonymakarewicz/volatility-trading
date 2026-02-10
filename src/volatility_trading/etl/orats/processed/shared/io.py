"""Shared IO/path helpers for processed ORATS datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _intermediate_endpoint_part_path(
    inter_api_root: Path | str,
    *,
    endpoint: str,
    ticker: str,
) -> Path:
    """Return intermediate path for one endpoint/ticker parquet partition."""
    root = Path(inter_api_root)
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("ticker must be non-empty")
    ep = str(endpoint).strip()
    if not ep:
        raise ValueError("endpoint must be non-empty")
    return root / f"endpoint={ep}" / f"underlying={t}" / "part-0000.parquet"


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
    path = _intermediate_endpoint_part_path(
        inter_api_root, endpoint=endpoint, ticker=ticker
    )
    if not path.exists():
        raise FileNotFoundError(
            f"API intermediate not found for endpoint={endpoint!r} "
            f"ticker={ticker!r}: {path}"
        )

    logger.info(
        "Reading API intermediate endpoint=%s ticker=%s: %s", endpoint, ticker, path
    )
    return pl.scan_parquet(str(path))


def processed_underlying_part_path(proc_root: Path | str, ticker: str) -> Path:
    """Return `proc_root/underlying=<TICKER>/part-0000.parquet`."""
    root = Path(proc_root)
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("ticker must be non-empty")
    return root / f"underlying={t}" / "part-0000.parquet"
