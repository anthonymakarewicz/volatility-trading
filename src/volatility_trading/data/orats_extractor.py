from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import zipfile

import polars as pl


def extract_ticker_from_orats(
    raw_root: str | Path,
    out_root: str | Path,
    ticker: str,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    *,
    root_col: str = "root",
    verbose: bool = True,
) -> None:
    """
    Extract a single underlying (ticker) from raw ORATS SMV Strikes ZIP files
    and store it as Parquet partitioned by year.

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

    Parameters
    ----------
    raw_root : str | Path
        Root directory containing the downloaded ORATS ZIPs.
    out_root : str | Path
        Root directory where by-ticker Parquet files will be written.
    ticker : str
        Underlying root symbol to extract (e.g. "SPX", "SPY").
    year_whitelist : Iterable[int] | Iterable[str] | None, optional
        If given, restrict extraction to these years.
    root_col : str, optional
        Name of the column in the ORATS CSV that contains the underlying root,
        by default "root".
    verbose : bool, optional
        If True, print progress messages, by default True.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    if year_whitelist is not None:
        year_whitelist_str = {str(y) for y in year_whitelist}
    else:
        year_whitelist_str = None

    if verbose:
        print(f"\n=== Extracting ticker: {ticker} ===")
        print(f"Raw root: {raw_root}")
        print(f"Out root: {out_root}")

    # Iterate over base dirs: smvstrikes_2007_2012, smvstrikes, ...
    for base_dir in sorted(raw_root.iterdir()):
        if not base_dir.is_dir():
            continue

        if verbose:
            print(f"\nBase directory: {base_dir.name}")

        # Iterate over year subdirectories
        for year_dir in sorted(base_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            year_name = year_dir.name
            if not year_name.isdigit():
                continue

            if year_whitelist_str is not None and year_name not in year_whitelist_str:
                continue

            if verbose:
                print(f"  Year {year_name} ...")

            dfs: list[pl.DataFrame] = []

            # One ZIP per trading day
            for zip_path in sorted(year_dir.glob("*.zip")):
                if verbose:
                    print(f"    Reading {zip_path.name} ...")

                with zipfile.ZipFile(zip_path) as zf:
                    # We expect exactly one CSV inside each ZIP
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pl.read_csv(f)

                # Filter to this ticker
                df_ticker = df.filter(pl.col(root_col) == ticker)

                if df_ticker.height > 0:
                    dfs.append(df_ticker)

            if not dfs:
                if verbose:
                    print(f"    No rows for {ticker} in {year_name}, skipping.")
                continue

            out_df = pl.concat(dfs, how="vertical").rechunk()

            out_dir = out_root / f"underlying={ticker}" / f"year={year_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "part-0000.parquet"

            if verbose:
                print(f"    Writing {out_path} (rows={out_df.height}, cols={out_df.width})")

            out_df.write_parquet(out_path)

    if verbose:
        print(f"\nFinished extracting {ticker}.")


def extract_tickers_from_orats(
    raw_root: str | Path,
    out_root: str | Path,
    tickers: Sequence[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    *,
    root_col: str = "root",
    verbose: bool = True,
) -> None:
    """
    Extract multiple tickers sequentially using `extract_ticker_from_orats`.

    Parameters
    ----------
    raw_root : str | Path
        Root directory containing the downloaded ORATS ZIPs.
    out_root : str | Path
        Root directory where by-ticker Parquet files will be written.
    tickers : Sequence[str]
        Collection of underlying roots to extract.
    year_whitelist : Iterable[int] | Iterable[str] | None, optional
        If given, restrict extraction to these years.
    root_col : str, optional
        Name of the column in the ORATS CSV that contains the underlying root,
        by default "root".
    verbose : bool, optional
        If True, print progress messages, by default True.
    """
    for ticker in tickers:
        extract_ticker_from_orats(
            raw_root=raw_root,
            out_root=out_root,
            ticker=ticker,
            year_whitelist=year_whitelist,
            root_col=root_col,
            verbose=verbose,
        )