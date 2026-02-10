"""Shared ORATS API IO helpers.

This module provides small dependency-light utilities used by the ORATS API
download/extract pipeline:
- year whitelist validation
- raw snapshot naming conventions
- raw/intermediate path builders
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path

MIN_YEAR: int = 2007

# Raw snapshot compression
DEFAULT_COMPRESSION: str = "gz"  # "gz" or "none"
ALLOWED_COMPRESSIONS: set[str] = {"gz", "none"}


def ensure_dir(p: Path) -> None:
    """Ensure a directory exists, creating parents as needed.

    Args:
        p: Directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def validate_years(
    years: Iterable[int | str],
    *,
    min_year: int = MIN_YEAR,
    max_year: int | None = None,
) -> list[int]:
    """Validate and coerce a year iterable into a list of integers.

    Args:
        years: Iterable of years (ints or strings).
        min_year: Minimum allowed year (inclusive).
        max_year: Maximum allowed year (inclusive). Defaults to current year.

    Returns:
        Validated years as integers.

    Raises:
        ValueError: If the input is empty or contains out-of-range years.
    """
    if max_year is None:
        max_year = dt.date.today().year

    years_list = [int(y) for y in years]
    if not years_list:
        raise ValueError("years must be non-empty")

    bad = [y for y in years_list if y < min_year or y > max_year]
    if bad:
        raise ValueError(
            f"Invalid years {bad}. Expected range [{min_year}, {max_year}]."
        )

    return years_list


def json_suffix(compression: str) -> str:
    """Return the raw snapshot file suffix for a compression mode.

    Args:
        compression: `"gz"` for gzip JSON or `"none"` for plain JSON.

    Returns:
        File suffix (`".json.gz"` or `".json"`).

    Raises:
        ValueError: If compression mode is unsupported.
    """
    if compression == "gz":
        return ".json.gz"
    if compression == "none":
        return ".json"
    raise ValueError(
        f"Unsupported compression '{compression}'. Allowed: "
        f"{sorted(ALLOWED_COMPRESSIONS)}"
    )


def raw_by_trade_date_dir(raw_root: Path, endpoint: str, year: int) -> Path:
    """Return BY_TRADE_DATE raw directory for one endpoint year.

    Layout: `raw_root/endpoint=<endpoint>/year=YYYY/`.
    """
    return raw_root / f"endpoint={endpoint}" / f"year={year}"


def raw_full_history_dir(raw_root: Path, endpoint: str, ticker: str) -> Path:
    """Return FULL_HISTORY raw directory for one endpoint/ticker pair.

    Layout: `raw_root/endpoint=<endpoint>/underlying=<TICKER>/`.
    """
    return raw_root / f"endpoint={endpoint}" / f"underlying={ticker}"


def raw_path_by_trade_date(
    raw_root: Path,
    endpoint: str,
    trade_date: str,
    part: int,
    compression: str,
) -> Path:
    """Return raw file path for one BY_TRADE_DATE request snapshot.

    Args:
        raw_root: Raw root directory.
        endpoint: Endpoint name.
        trade_date: Trade date in `YYYY-MM-DD`.
        part: Ticker chunk index (`chunkXXX`).
        compression: Compression mode (`"gz"` or `"none"`).

    Returns:
        Full snapshot path under `endpoint=<endpoint>/year=<YYYY>/`.
    """
    year = int(trade_date[:4])
    base = raw_by_trade_date_dir(raw_root, endpoint, year)
    return base / f"{trade_date}_chunk{part:03d}{json_suffix(compression)}"


def raw_path_full_history(
    raw_root: Path,
    endpoint: str,
    ticker: str,
    compression: str,
) -> Path:
    """Return raw file path for one FULL_HISTORY snapshot.

    Layout: `raw_root/endpoint=<endpoint>/underlying=<TICKER>/data.json(.gz)`.
    """
    base = raw_full_history_dir(raw_root, endpoint, ticker)
    return base / f"data{json_suffix(compression)}"


def intermediate_full_history(
    intermediate_root: Path,
    endpoint: str,
    ticker: str,
) -> Path:
    """Return default intermediate parquet path for one endpoint/ticker.

    Layout:
        `intermediate_root/endpoint=<endpoint>/underlying=<TICKER>/part-0000.parquet`.
    """
    return (
        intermediate_root
        / f"endpoint={endpoint}"
        / f"underlying={ticker}"
        / "part-0000.parquet"
    )
