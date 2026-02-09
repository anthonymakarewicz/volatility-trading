from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from pandas.errors import EmptyDataError
from pandas.tseries.offsets import BMonthEnd


def extract_7z_and_load(
    archive_path: str | Path, low_memory: bool = True
) -> pd.DataFrame:
    """
    Extract a single .7z archive, read all non-empty .txt files inside,
    and return a concatenated DataFrame of their contents.

    Parameters
    ----------
    archive_path : str | Path
        Path to the .7z archive file.
    low_memory : bool, optional
        Passed to pandas.read_csv for memory optimization, by default True

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all non-empty .txt files inside the archive.

    Raises
    ------
    EmptyDataError
        If no non-empty .txt files are found or all are empty.
    """
    archive_path = Path(archive_path)

    extract_root_dir = archive_path.parent / "_tmp_extract"
    extract_dir = extract_root_dir / archive_path.stem

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)

    all_dfs = []
    for file in extract_dir.glob("*.txt"):
        if file.stat().st_size == 0:
            print(f"Skipping empty file: {file}")
            continue
        try:
            print(f"Reading {file}")
            df = pd.read_csv(file, low_memory=low_memory)
            all_dfs.append(df)
        except EmptyDataError:
            print(
                f"Warning: EmptyDataError encountered while reading {file}, skipping."
            )

    shutil.rmtree(extract_dir, ignore_errors=True)

    if not all_dfs:
        raise EmptyDataError(
            f"No non-empty .txt files found in archive: {archive_path}"
        )

    return pd.concat(all_dfs, ignore_index=True)


def load_options(
    root: str | Path,
    start_year: int = 2012,
    end_year: int = 2022,
    low_memory: bool = True,
) -> pd.DataFrame:
    """
    Walk year subdirectories under `root`, load all .7z archives between start_year and end_year,
    and return a raw concatenated DataFrame with normalized column names.

    Parameters
    ----------
    root : str | Path
        Root directory containing year subdirectories with .7z archives.
    start_year : int, optional
        Start year (inclusive) to load data from, by default 2012
    end_year : int, optional
        End year (inclusive) to load data from, by default 2022
    low_memory : bool, optional
        Passed to pandas.read_csv for memory optimization, by default True

    Returns
    -------
    pd.DataFrame
        Concatenated raw options data with normalized columns.

    Raises
    ------
    FileNotFoundError
        If no .7z archives found in the specified year range.
    """
    raw_root_dir = Path(root)
    frames = []

    for year_dir in sorted(raw_root_dir.glob("*")):
        name = year_dir.name
        if not name.isdigit():
            continue

        year = int(name)
        if not (start_year <= year <= end_year):
            continue

        print(f"Processing year: {year}")
        for archive in sorted(year_dir.glob("*.7z")):
            print(f"  ➤ Extracting {archive.name}")
            df = extract_7z_and_load(archive, low_memory=low_memory)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No .7z archives found between years {start_year} and {end_year} in {root}"
        )

    df = pd.concat(frames, ignore_index=True)

    df.columns = (
        df.columns.str.strip().str.replace(r"[\[\]]", "", regex=True).str.lower()
    )
    df = df.rename(columns={"quote_date": "date", "expire_date": "expiry"})

    return df


def reshape_options_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide options DataFrame with separate call (c_*) and put (p_*) columns
    into a long format with a single set of columns and an 'option_type' column.

    Parameters
    ----------
    df : pd.DataFrame
        Wide options DataFrame containing columns starting with 'c_' and 'p_'.

    Returns
    -------
    pd.DataFrame
        Long format DataFrame with an 'option_type' column indicating 'C' or 'P'.
    """
    df = df.copy()

    call_cols = [col for col in df.columns if col.startswith("c_")]
    put_cols = [col for col in df.columns if col.startswith("p_")]
    shared_cols = [col for col in df.columns if col not in call_cols + put_cols]

    call_rows = df[shared_cols + call_cols].copy()
    call_rows = call_rows.rename(columns={col: col[2:] for col in call_cols})
    call_rows["option_type"] = "C"

    put_rows = df[shared_cols + put_cols].copy()
    put_rows = put_rows.rename(columns={col: col[2:] for col in put_cols})
    put_rows["option_type"] = "P"

    long_df = pd.concat([call_rows, put_rows], axis=0)
    long_df = long_df.sort_values(["date", "strike", "option_type"])

    return long_df


def reshape_options_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long options DataFrame (with 'option_type' column) into a wide format
    with separate c_* and p_* columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long format options DataFrame containing 'option_type' column.

    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with separate call and put columns prefixed by 'c_' and 'p_'.
    """
    df = df.copy()

    if "option_type" not in df.columns:
        raise ValueError("Expected column 'option_type' not found.")

    if df.index.name == "date":
        df = df.reset_index()

    index_cols = [
        "date",
        "underlying_last",
        "expiry",
        "dte",
        "strike",
        "strike_distance",
        "strike_distance_pct",
    ]

    index_cols = [col for col in df.columns if col in index_cols]
    wide = df.pivot(index=index_cols, columns="option_type")
    wide.columns = [f"{opt.lower()}_{col.lower()}" for col, opt in wide.columns]

    if "date" in index_cols:
        wide = wide.reset_index().set_index("date").sort_index()
    else:
        wide = wide.reset_index()

    return wide


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw options data by reshaping from wide to long format, dropping unused columns,
    parsing dates, enforcing numeric types, and performing basic sanity checks on prices and DTE.

    Parameters
    ----------
    df : pd.DataFrame
        Raw options DataFrame in wide format.

    Returns
    -------
    pd.DataFrame
        Cleaned long format options DataFrame indexed by date.
    """
    df = df.copy()

    df = reshape_options_wide_to_long(df)

    cols_to_drop = [
        "quote_unixtime",
        "quote_readtime",
        "quote_time_hours",
        "expire_unix",
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    size_split = (
        df["size"]
        .astype(str)
        .str.extract(r"(?P<size_bid>\d+)\s*[xX]\s*(?P<size_ask>\d+)")
    )
    df["size_bid"] = pd.to_numeric(size_split["size_bid"], errors="coerce")
    df["size_ask"] = pd.to_numeric(size_split["size_ask"], errors="coerce")
    df = df.drop("size", axis=1)

    # Convert date columns
    dt_cols = ["date", "expiry"]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.set_index("date")
    df = df.sort_index()
    df = df.sort_values(by=["date", "strike"])

    # Convert numeric columns
    numeric_cols = set(df.columns) - set(dt_cols + ["option_type"])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)

    # Ensure bid/ask/last >= 0 and ask >= bid
    df = df[df["ask"] >= df["bid"]]
    for col in ["bid", "ask", "last"]:
        df = df[df[col] >= 0]

    # Ensure non-negative DTE
    df = df[df["dte"] >= 0]
    df["dte"] = np.round(df["dte"])  # Avoid decimal expiries

    return df


def remove_illiquid_options(
    df: pd.DataFrame,
    volume_min: int = 1,
    rel_spread_max: float = 0.25,
    moneyness_band: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """
    Filter out illiquid options based on minimum volume, relative bid-ask spread,
    positive bid and ask prices, and moneyness band.

    Parameters
    ----------
    df : pd.DataFrame
        Options DataFrame to filter.
    volume_min : int, optional
        Minimum volume threshold, by default 1
    rel_spread_max : float, optional
        Maximum allowed relative bid-ask spread, by default 0.25
    moneyness_band : tuple[float, float], optional
        Allowed range of strike / underlying price ratio, by default (0.8, 1.2)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only liquid options.
    """
    df = df.copy()

    # Drop bad prices
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]

    # Check for wide bid-ask spread
    mid = 0.5 * (df["bid"] + df["ask"])
    df = df[((df["ask"] - df["bid"]) / mid) <= rel_spread_max]

    # Remove low volumes
    df = df[df["volume"] >= volume_min]

    moneyness = df["strike"] / df["underlying_last"]
    df = df[(moneyness >= moneyness_band[0]) & (moneyness <= moneyness_band[1])]

    return df


def extract_eom_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame indexed by date, return only rows where the expiry date is the month-end.

    Parameters
    ----------
    df : pd.DataFrame
        Options DataFrame indexed by a DateTimeIndex.

    Returns
    -------
    pd.DataFrame
        Subset of the input DataFrame where expiry is the month-end.
    """
    df = df.copy()

    df["eom_expiry"] = df.index.to_series().add(BMonthEnd(0)).values
    eom_options = df[df["eom_expiry"] == df["expiry"]].drop(columns="eom_expiry")

    return eom_options


def prepare_option_panel(
    *,
    raw_root: str | Path,
    output_path: str | Path,
    start_year: int = 2012,
    end_year: int = 2022,
    low_memory: bool = False,
    do_clean: bool = True,
    reshape: str | None = "wide",
    verbose: bool = True,
    return_df: bool = False,
) -> Path | pd.DataFrame:
    """
    End-to-end helper to load raw .7z files from raw_root/[year]/, optionally clean and reshape,
    and save a single Parquet file at output_path.

    Parameters
    ----------
    raw_root : str | Path
        Root directory containing yearly subdirectories of .7z archives.
    output_path : str | Path
        Path where the output Parquet file will be saved.
    start_year : int, optional
        First year (inclusive) to load data from, by default 2012
    end_year : int, optional
        Last year (inclusive) to load data from, by default 2022
    low_memory : bool, optional
        Passed to pandas.read_csv for memory optimization, by default False
    do_clean : bool, optional
        Whether to clean the raw loaded data, by default True
    reshape : str | None, optional
        Reshaping mode after cleaning:
          - "wide": reshape long to wide,
          - "long": keep long format,
          - None: no reshaping,
        by default "wide"
    verbose : bool, optional
        Whether to print progress messages, by default True
    return_df : bool, optional
        If True, return the resulting DataFrame instead of the output path, by default False

    Returns
    -------
    Path | pd.DataFrame
        Path to the saved Parquet file or the DataFrame if return_df is True.

    Raises
    ------
    ValueError
        If an invalid reshape argument is provided.
    """
    raw_root = Path(raw_root)
    output_path = Path(output_path)

    if verbose:
        print(
            f"Loading options data from {raw_root} for years {start_year} to {end_year}..."
        )

    df = load_options(
        raw_root, start_year=start_year, end_year=end_year, low_memory=low_memory
    )

    if verbose:
        print(f"Loaded data shape: {df.shape}")

    if do_clean:
        if verbose:
            print("Cleaning data...")
        df = clean_data(df)
        if verbose:
            print(f"Data shape after cleaning: {df.shape}")

    if reshape == "wide":
        if verbose:
            print("Reshaping to wide format...")
        df = reshape_options_long_to_wide(df)
    elif reshape == "long":
        if verbose:
            print("Keeping long format...")
        pass  # keep as long format
    elif reshape is None:
        if verbose:
            print("Skipping reshaping...")
        pass  # no reshaping
    else:
        raise ValueError(
            f"Invalid reshape option '{reshape}'. Expected 'wide', 'long', or None."
        )

    if verbose:
        print(f"Writing Parquet to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    if verbose:
        print("Finished writing panel.")

    if return_df:
        return df
    return output_path
