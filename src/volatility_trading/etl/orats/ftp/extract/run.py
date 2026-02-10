"""Extraction step for ORATS FTP ZIP snapshots.

Reads raw ZIP snapshots, normalizes vendor columns, filters requested tickers,
and writes partitioned intermediate parquet outputs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl
from polars.exceptions import NoDataError

from ..types import ExtractFtpResult
from ._helpers import read_orats_zip_to_polars

logger = logging.getLogger(__name__)

ROOT_COL: str = "ticker"


def extract(
    *,
    raw_root: str | Path,
    out_root: str | Path,
    tickers: Sequence[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    strict: bool = True,
) -> ExtractFtpResult:
    """Extract ORATS FTP snapshots into normalized intermediate parquet data.

    Args:
        raw_root: Root directory containing raw FTP ZIP dumps.
        out_root: Output root directory for parquet partitions.
        tickers: Underlying tickers to extract (for example `["SPX", "AAPL"]`).
        year_whitelist: Optional allowlist of years to read.
        strict: If `True`, raise when one or more ZIP files cannot be read.

    Returns:
        Summary including read/write counts, output paths, and failures.

    Raises:
        RuntimeError: If `strict=True` and at least one ZIP fails to read.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    tickers = list(tickers)

    t0 = time.perf_counter()
    out_paths: list[Path] = []
    failed_paths: list[Path] = []

    n_seen = 0
    n_read = 0
    n_failed = 0

    n_rows_before = 0
    n_rows_after = 0
    n_dupes_dropped = 0

    if year_whitelist is not None:
        year_whitelist_str = {str(y) for y in year_whitelist}
    else:
        year_whitelist_str = None

    logger.info("=== Extracting tickers: %s ===", ", ".join(tickers))
    logger.info("Raw root: %s", raw_root)
    logger.info("Out root: %s", out_root)

    # iterate over base dirs: smvstrikes_2007_2012, smvstrikes, ...
    for base_dir in sorted(raw_root.iterdir()):
        if not base_dir.is_dir():
            continue

        logger.info("Base directory: %s", base_dir.name)

        # iterate over year subdirectories
        for year_dir in sorted(base_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            year_name = year_dir.name
            if not year_name.isdigit():
                continue

            if year_whitelist_str is not None and year_name not in year_whitelist_str:
                continue

            logger.info("Year %s ...", year_name)

            # per-year accumulator: ticker -> list of DataFrames
            dfs_by_ticker: dict[str, list[pl.DataFrame]] = {t: [] for t in tickers}

            # loop over daily ZIP files in this year
            for zip_path in sorted(year_dir.glob("*.zip")):
                n_seen += 1
                logger.debug("Reading %s ...", zip_path.name)
                try:
                    df = read_orats_zip_to_polars(zip_path)
                    n_read += 1
                except (FileNotFoundError, NoDataError) as e:
                    n_failed += 1
                    failed_paths.append(zip_path)
                    logger.error("[ERROR] %s: %s", zip_path, e)
                    continue

                # filter once per ticker, reusing the same df
                for ticker in tickers:
                    df_ticker = df.filter(pl.col(ROOT_COL) == ticker)
                    if df_ticker.height > 0:
                        dfs_by_ticker[ticker].append(df_ticker)

            # after processing all ZIPs for this year, write each ticker's data
            for ticker in tickers:
                dfs = dfs_by_ticker[ticker]
                if not dfs:
                    logger.info(
                        "No rows for %s in %s, skipping.",
                        ticker,
                        year_name,
                    )
                    continue

                out_df = pl.concat(dfs, how="vertical").rechunk()
                before = out_df.height

                # De-duplicate exact duplicate rows (best-effort stable order)
                out_df = out_df.unique(maintain_order=True)
                after = out_df.height

                n_rows_before += before
                n_rows_after += after
                dropped = before - after
                if dropped:
                    n_dupes_dropped += dropped
                    logger.info(
                        "De-duplicated rows for ticker=%s year=%s: %d ->"
                        " %d (dropped=%d)",
                        ticker,
                        year_name,
                        before,
                        after,
                        dropped,
                    )

                out_dir = out_root / f"underlying={ticker}" / f"year={year_name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "part-0000.parquet"

                logger.info(
                    "Writing %s (ticker=%s, rows=%d, cols=%d)",
                    out_path,
                    ticker,
                    out_df.height,
                    out_df.width,
                )

                out_df.write_parquet(out_path)
                out_paths.append(out_path)

    result = ExtractFtpResult(
        n_zip_files_seen=n_seen,
        n_zip_files_read=n_read,
        n_zip_files_failed=n_failed,
        n_rows_total_before_dedup=n_rows_before,
        n_rows_total_after_dedup=n_rows_after,
        n_duplicates_dropped=n_dupes_dropped,
        n_out_files=len(out_paths),
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed_paths,
    )

    logger.info(
        "Finished FTP extract seen=%d read=%d "
        "written=%d failed=%d rows=%d duration_s=%.2f",
        result.n_zip_files_seen,
        result.n_zip_files_read,
        result.n_out_files,
        result.n_zip_files_failed,
        result.n_rows_total_after_dedup,
        result.duration_s,
    )

    if strict and result.n_zip_files_failed:
        raise RuntimeError(
            f"Extraction finished with {result.n_zip_files_failed} "
            "problematic ZIP files."
        )

    return result
