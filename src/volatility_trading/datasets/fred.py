"""I/O helpers for processed FRED domain datasets.

Processed FRED data is stored by domain:
- `<proc_root>/<domain>/fred_<domain>.parquet` when using source root
- `<domain_root>/fred_<domain>.parquet` for domain-specific roots
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_FRED, PROC_FRED_MARKET, PROC_FRED_RATES

FRED_INDEX_COL = "__index_level_0__"


def _normalize_domain(domain: str) -> str:
    normalized = str(domain).strip().lower()
    if not normalized:
        raise ValueError("domain must be non-empty")
    return normalized


def _normalize_fred_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    columns = set(schema.names())
    if FRED_INDEX_COL in columns and "date" not in columns:
        lf = lf.rename({FRED_INDEX_COL: "date"})
    return lf


def fred_domain_path(domain: str, *, proc_root: Path | str = PROC_FRED) -> Path:
    """Build a processed FRED parquet path from the source-level processed root."""
    d = _normalize_domain(domain)
    root = Path(proc_root)
    return root / d / f"fred_{d}.parquet"


def scan_fred_domain(
    domain: str,
    *,
    proc_root: Path | str = PROC_FRED,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan one processed FRED domain parquet lazily from source-level root."""
    path = fred_domain_path(domain, proc_root=proc_root)
    if not path.exists():
        raise FileNotFoundError(f"Processed FRED domain dataset not found: {path}")
    lf = _normalize_fred_schema(pl.scan_parquet(path))
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_fred_domain(
    domain: str,
    *,
    proc_root: Path | str = PROC_FRED,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read one processed FRED domain parquet eagerly from source-level root."""
    return scan_fred_domain(domain, proc_root=proc_root, columns=columns).collect()


def fred_rates_path(proc_root: Path | str = PROC_FRED_RATES) -> Path:
    """Build the processed FRED rates parquet path from a rates domain root."""
    return Path(proc_root) / "fred_rates.parquet"


def scan_fred_rates(
    *,
    proc_root: Path | str = PROC_FRED_RATES,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan processed FRED rates parquet lazily from rates domain root."""
    path = fred_rates_path(proc_root)
    if not path.exists():
        raise FileNotFoundError(f"Processed FRED rates dataset not found: {path}")
    lf = _normalize_fred_schema(pl.scan_parquet(path))
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_fred_rates(
    *,
    proc_root: Path | str = PROC_FRED_RATES,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read processed FRED rates parquet eagerly from rates domain root."""
    return scan_fred_rates(proc_root=proc_root, columns=columns).collect()


def fred_market_path(proc_root: Path | str = PROC_FRED_MARKET) -> Path:
    """Build the processed FRED market parquet path from a market domain root."""
    return Path(proc_root) / "fred_market.parquet"


def scan_fred_market(
    *,
    proc_root: Path | str = PROC_FRED_MARKET,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan processed FRED market parquet lazily from market domain root."""
    path = fred_market_path(proc_root)
    if not path.exists():
        raise FileNotFoundError(f"Processed FRED market dataset not found: {path}")
    lf = _normalize_fred_schema(pl.scan_parquet(path))
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_fred_market(
    *,
    proc_root: Path | str = PROC_FRED_MARKET,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read processed FRED market parquet eagerly from market domain root."""
    return scan_fred_market(proc_root=proc_root, columns=columns).collect()
