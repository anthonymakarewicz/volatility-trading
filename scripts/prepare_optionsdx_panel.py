#!/usr/bin/env python
"""
Build a cleaned OptionsDX options panel from raw .7z archives.

Expected layout:
    data/raw/optionsdx/<YEAR>/*.7z

Output:
    data/processed/optionsdx/optionsdx_panel_<START>_<END>.parquet
"""
from pathlib import Path
from volatility_trading.data import prepare_option_panel


# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

RAW_OPTIONSDX_ROOT = DATA_ROOT / "raw" / "optionsdx"
PROCESSED_OPTIONSDX_ROOT = DATA_ROOT / "processed" / "optionsdx"

START_YEAR = 2010
END_YEAR = 2023
LOW_MEMORY = False  # set True if you hit memory issues

PANEL_NAME = f"optionsdx_panel_{START_YEAR}_{END_YEAR}.parquet"
OUTPUT_PATH = PROCESSED_OPTIONSDX_ROOT / PANEL_NAME


def main() -> None:
    print(f"Raw root:       {RAW_OPTIONSDX_ROOT}")
    print(f"Output path:    {OUTPUT_PATH}")
    print(f"Years:          {START_YEAR}–{END_YEAR}")
    print(f"low_memory:     {LOW_MEMORY}")
    PROCESSED_OPTIONSDX_ROOT.mkdir(parents=True, exist_ok=True)

    panel_path = prepare_option_panel(
        raw_root=RAW_OPTIONSDX_ROOT,
        output_path=OUTPUT_PATH,
        start_year=START_YEAR,
        end_year=END_YEAR,
        low_memory=LOW_MEMORY,
        do_clean=True,
        reshape="wide",       # or "long" / None
        return_df=False,      # set True if you want the df in memory
    )

    print(f"\nDone. Panel saved at: {panel_path}")


if __name__ == "__main__":
    main()