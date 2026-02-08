#!/usr/bin/env python
"""
Build a cleaned OptionsDX options panel from raw .7z archives.

Expected layout:
    data/raw/optionsdx/<YEAR>/*.7z

Output:
    data/processed/optionsdx/optionsdx_panel_<START>_<END>.parquet
"""
from volatility_trading.config.paths import PROC_OPTIONSDX
from volatility_trading.etl.optionsdx_loader import prepare_option_panel

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

START_YEAR = 2010
END_YEAR = 2023
LOW_MEMORY = False  # set True if you hit memory issues

PANEL_NAME = f"optionsdx_panel_{START_YEAR}_{END_YEAR}.parquet"
OUTPUT_PATH = PROC_OPTIONSDX / PANEL_NAME


def main() -> None:
    print(f"Raw root:       {PROC_OPTIONSDX}")
    print(f"Output path:    {OUTPUT_PATH}")
    print(f"Years:          {START_YEAR}–{END_YEAR}")
    print(f"low_memory:     {LOW_MEMORY}")
    PROC_OPTIONSDX.mkdir(parents=True, exist_ok=True)

    panel_path = prepare_option_panel(
        raw_root=PROC_OPTIONSDX,
        output_path=OUTPUT_PATH,
        start_year=START_YEAR,
        end_year=END_YEAR,
        low_memory=LOW_MEMORY,
        do_clean=True,
        reshape="wide",       # or "long" / None
        verbose=True,
        return_df=False,      # set True if you want the df in memory
    )

    print(f"\nDone. Panel saved at: {panel_path}")


if __name__ == "__main__":
    main()
