import py7zr
import pandas as pd
import shutil
import numpy as np
import os

from pathlib import Path
from pandas.tseries.offsets import BMonthEnd


def extract_7z_and_load(csv_path):
    with py7zr.SevenZipFile(csv_path, mode='r') as archive:
        archive.extractall(path='tmp')

    extracted_files = list(Path('tmp').glob("*"))
    all_dfs = []
    for file in extracted_files:

        print(f"Reading {file}")
        df = pd.read_csv(file, sep=",")
        all_dfs.append(df)

    shutil.rmtree('tmp', ignore_errors=True)
    return pd.concat(all_dfs, ignore_index=True)


def load_options(root, start_year=2012, end_year=2022):
    raw_root = Path(root)
    all_dfs = []
    #os.chdir(root)
    #os.remove(".DS_store")

    for year_dir in sorted(raw_root.glob("*")):
        name = year_dir.name
        if not name.isdigit():
            continue    

        year = int(name)
        if not (start_year <= year <= end_year):
            continue
        
        print(f"Processing year: {year}")
        for archive in sorted(year_dir.glob("*.7z")):
            print(f"  ➤ Extracting {archive.name}")
            df = extract_7z_and_load(archive)
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\[\]]", "", regex=True)
        .str.lower()
    )
    df = df.rename(columns={
        "quote_date": "date",
        "expire_date": "expiry"
    })

    return df


def reshape_options_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    call_cols = [col for col in df.columns if col.startswith("c_")]
    put_cols = [col for col in df.columns if col.startswith("p_")]
    shared_cols = [col for col in df.columns if col not in call_cols + put_cols]

    df_calls = df[shared_cols + call_cols].copy()
    df_calls = df_calls.rename(columns={col: col[2:] for col in call_cols})
    df_calls['option_type'] = 'C'

    df_puts = df[shared_cols + put_cols].copy()
    df_puts = df_puts.rename(columns={col: col[2:] for col in put_cols})
    df_puts['option_type'] = 'P'

    df_long = pd.concat([df_calls, df_puts], axis=0)
    df_long = df_long.sort_values(["date", "strike", "option_type"])

    return df_long


def reshape_options_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "option_type" not in df.columns:
        raise ValueError("Expected column 'option_type' not found.")

    if df.index.name == "date":
        df = df.reset_index()

    cols_to_include = [
        'date',
        'underlying_last',
        'expiry',
        'dte',
        'strike',
        'strike_distance',
        'strike_distance_pct'
    ]

    non_option_cols = [col for col in df.columns if col in cols_to_include]
    wide_df = df.pivot(index=non_option_cols, columns="option_type")
    wide_df.columns = [f"{opt.lower()}_{col.lower()}" for col, opt in wide_df.columns]

    if "date" in non_option_cols:
        wide_df = wide_df.reset_index().set_index("date").sort_index()
    else:
        wide_df = wide_df.reset_index()

    return wide_df


def clean_data(df):
    df = df.copy()

    df = reshape_options_wide_to_long(df)

    cols_to_drop = ['quote_unixtime', 'quote_readtime', 'quote_time_hours', "expire_unix"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    size_split = df['size'].astype(str).str.extract(r'(?P<size_bid>\d+)\s*[xX]\s*(?P<size_ask>\d+)')
    df['size_bid'] = pd.to_numeric(size_split['size_bid'], errors='coerce')
    df['size_ask'] = pd.to_numeric(size_split['size_ask'], errors='coerce')
    df = df.drop('size', axis=1)

    # Convert date columns
    dt_cols = ["date", "expiry"]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.set_index("date")
    df = df.sort_index()
    df = df.sort_values(by=["date", "strike"])

    # Convert numeric columns
    num_cols = set(df.columns) - set(dt_cols + ["option_type"])
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaNs in critical fields and duplicates
    # critical_cols = ['iv', 'delta', 'bid', 'ask', 'last']
    # df = df.dropna(subset=critical_cols)
    # df = df.drop_duplicates()

    df["volume"] = df["volume"].fillna(0)

    # Ensure bid/ask/last >= 0
    df = df[df['ask'] >= df['bid']]
    for col in ["bid", "ask", "last"]:
        df = df[df[col] >= 0]

    # Ensure non-negative DTE
    df = df[df['dte'] >= 0]
    df["dte"] = np.round(df["dte"]) # Avoid decimal expiries

    return df


def remove_illiquid_options(df, volume_min=1, rel_spread_max=0.25, moneyness_band=(0.8, 1.2)):
    df = df.copy()

    # Drop bad prices
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]

    # Check for wide bid-ask spread
    mid = 0.5 * (df['bid'] + df['ask'])
    df = df[((df['ask'] - df['bid']) / mid) <= rel_spread_max]

    # Remove low volumes
    df = df[df['volume'] >= volume_min]

    moneyness = df['strike'] / df['underlying_last']
    df = df[(moneyness >= moneyness_band[0]) & (moneyness <= moneyness_band[1])]

    return df


def extract_eom_options(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["eom_expiry"] = (df.index + BMonthEnd(0)).to_list()
    eom_df = df[df["eom_expiry"] == df["expiry"]].drop(columns="eom_expiry")

    return eom_df


def extract_options():
    # TODO: Create a funciton that all the options's extraction in one fucntion call
    pass