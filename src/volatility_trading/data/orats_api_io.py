"""Shared ORATS API IO helpers.

This module contains small, dependency-light utilities used by the ORATS API
pipeline (downloader + extractor):

- Validate year whitelists (bounded by min_year and current year by default).
- Encode raw snapshot filename conventions (JSON vs JSON.GZ).
- Build raw/intermediate directory paths in a consistent layout.
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
    """Ensure a directory exists.

    Creates the directory and any missing parents. Safe to call repeatedly.

    Parameters
    ----------
    p:
        Directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def validate_years(
    years: Iterable[int | str],
    *,
    min_year: int = MIN_YEAR,
    max_year: int | None = None,
) -> list[int]:
    """Validate and coerce an iterable of years into a list of ints.

    Used by download/extract orchestration for endpoints that require an explicit
    year whitelist.

    Rules:
    - Input must be non-empty.
    - Years must be within [min_year, max_year].
    - max_year defaults to the current calendar year.

    Parameters
    ----------
    years:
        Iterable of years (ints or strings).
    min_year:
        Minimum allowed year (inclusive).
    max_year:
        Maximum allowed year (inclusive). Defaults to date.today().year.

    Returns
    -------
    list[int]
        Validated years as integers.

    Raises
    ------
    ValueError
        If the input is empty or contains out-of-range years.
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
    """Return the raw snapshot suffix for a compression mode.

    Parameters
    ----------
    compression:
        "gz" for gzip-compressed JSON, or "none" for plain JSON.

    Returns
    -------
    str
        File suffix (".json.gz" or ".json").

    Raises
    ------
    ValueError
        If the compression mode is unsupported.
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
    """Directory containing BY_TRADE_DATE raw snapshots for a year.

    Layout:
        raw_root/endpoint=<endpoint>/year=YYYY/
    """
    return raw_root / f"endpoint={endpoint}" / f"year={year}"


def raw_full_history_dir(raw_root: Path, endpoint: str, ticker: str) -> Path:
    """Directory containing FULL_HISTORY raw snapshots for one underlying.

    Layout:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/
    """
    return raw_root / f"endpoint={endpoint}" / f"underlying={ticker}"


def raw_path_by_trade_date(
    raw_root: Path,
    endpoint: str,
    trade_date: str,
    part: int,
    compression: str,
) -> Path:
    """Full path for one BY_TRADE_DATE raw snapshot file.

    Layout:
        raw_root/endpoint=<endpoint>/year=YYYY/
            YYYY-MM-DD_chunkXXX.json(.gz)

    Notes
    -----
    The `part` index corresponds to the ticker chunk (e.g. chunk000 for the
    first group of tickers).
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
    """Full path for one FULL_HISTORY raw snapshot file.

    Layout:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/data.json(.gz)
    """
    base = raw_full_history_dir(raw_root, endpoint, ticker)
    return base / f"data{json_suffix(compression)}"


def intermediate_full_history(
    intermediate_root: Path,
    endpoint: str,
    ticker: str,
) -> Path:
    """Default intermediate parquet path for one ticker (full history).

    Layout:
        intermediate_root/endpoint=<endpoint>/underlying=<TICKER>/full.parquet
    """
    return (
        intermediate_root
        / f"endpoint={endpoint}"
        / f"underlying={ticker}"
        / "full.parquet"
    )