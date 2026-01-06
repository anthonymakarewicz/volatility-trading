from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl
from polars.exceptions import NoDataError

from volatility_trading.config.orats_ftp_schemas import STRIKES_VENDOR_DTYPES

logger = logging.getLogger(__name__)

ROOT_COl: str = "ticker"


def _read_orats_zip_to_polars(zip_path: Path) -> pl.DataFrame:
    """
    Open an ORATS SMV Strikes ZIP and return a Polars DataFrame.

    Handles both layouts:
    - file.csv
    - some_folder/file.csv
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # keep only actual CSV files (ignore directories)
        csv_candidates = [
            name
            for name in names
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]

        if not csv_candidates:
            raise FileNotFoundError(
                f"No CSV file found inside {zip_path.name}; entries={names}"
            )

        csv_name = csv_candidates[0]

        with zf.open(csv_name) as f:
            df = pl.read_csv(
                f, 
                schema_overrides=STRIKES_VENDOR_DTYPES, 
                null_values=["NULL"]
            )

    return df


def extract_tickers_from_orats(
    *,
    raw_root: str | Path,
    out_root: str | Path,
    tickers: Sequence[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    root_col: str = ROOT_COl,
) -> None:
    """
    Extract multiple tickers from raw ORATS SMV Strikes ZIP files and
    store them as Parquet partitioned by year.

    Expected local layout (raw_root)
    --------------------------------
    raw_root/
        smvstrikes_2007_2012/
            2007/*.zip
            2008/*.zip
            ...
        smvstrikes/
            2013/*.zip
            2014/*.zip
            ...

    Output layout (out_root)
    ------------------------
    out_root/
        underlying=<TICKER>/year=<YYYY>/part-0000.parquet

    Notes
    -----
    - Each ZIP is read *once*, then filtered for all requested tickers.
      This is much faster than reading each ZIP once per ticker.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    tickers = list(tickers)

    if year_whitelist is not None:
        year_whitelist_str = {str(y) for y in year_whitelist}
    else:
        year_whitelist_str = None

    logger.info("=== Extracting tickers: %s ===", ", ".join(tickers))
    logger.info("Raw root: %s", raw_root)
    logger.info("Out root: %s", out_root)

    # collect errors once per ZIP (not per ticker)
    zip_errors: list[tuple[str, str]] = []  # (zip_path, message)

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

            if (
                year_whitelist_str is not None and 
                year_name not in year_whitelist_str
            ):
                continue

            logger.info("Year %s ...", year_name)

            # per-year accumulator: ticker -> list of DataFrames
            dfs_by_ticker: dict[str, list[pl.DataFrame]] = {
                t: [] for t in tickers
            }

            # loop over daily ZIP files in this year
            for zip_path in sorted(year_dir.glob("*.zip")):
                logger.debug("Reading %s ...", zip_path.name)
                try:
                    df = _read_orats_zip_to_polars(zip_path)
                except (FileNotFoundError, NoDataError) as e:
                    zip_errors.append((str(zip_path), str(e)))
                    logger.error("[ERROR] %s: %s", zip_path, e)
                    continue

                # filter once per ticker, reusing the same df
                for ticker in tickers:
                    df_ticker = df.filter(pl.col(root_col) == ticker)
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

    # after all base dirs / years
    if zip_errors:
        logger.error("=== ERRORS DURING EXTRACTION ===")
        for path, msg in zip_errors:
            logger.error("- %s: %s", path, msg)
        raise RuntimeError(
            f"Extraction finished with {len(zip_errors)} problematic ZIP files."
        )

    logger.info("Finished extracting tickers: %s", ", ".join(tickers))


def extract_ticker_from_orats(
    *,
    raw_root: str | Path,
    out_root: str | Path,
    ticker: str,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    root_col: str = ROOT_COl,
) -> None:
    """
    Convenience wrapper around `extract_tickers_from_orats` for a single ticker.
    """
    extract_tickers_from_orats(
        raw_root=raw_root,
        out_root=out_root,
        tickers=[ticker],
        year_whitelist=year_whitelist,
        root_col=root_col,
    )