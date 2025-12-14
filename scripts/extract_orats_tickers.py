#!/usr/bin/env python
"""
Extract selected underlyings from raw ORATS SMV Strikes ZIP files
into by-ticker Parquet files.

Expected local layout (after running download_orats_raw.py):
    data/raw/orats/smvstrikes_2007_2012/2007/*.zip
    data/raw/orats/smvstrikes_2007_2012/2008/*.zip
    data/raw/orats/smvstrikes/2013/*.zip
    ...

Output layout (this script):
    data/intermediate/orats/by_ticker/
        underlying=SPX/year=2013/part-0000.parquet
        underlying=SPX/year=2014/part-0000.parquet
        underlying=SPY/year=2013/part-0000.parquet
        ...

Usage:
    python scripts/extract_orats_tickers.py
"""
from volatility_trading.config.paths import RAW_ORATS, INTER_ORATS_BY_TICKER
from volatility_trading.data import extract_tickers_from_orats

# ---- CONFIG ----

RAW_ORATS_ROOT = RAW_ORATS
OUT_ROOT = INTER_ORATS_BY_TICKER

# Choose which tickers you want to extract
TICKERS = ["SPY"]  # e.g. ["SPX", "SPY", "QQQ"]

# Restrict to specific years if you want (ints or strings), or None for all:
# YEAR_WHITELIST = {2013, 2014}
YEAR_WHITELIST = [2024, 2025]

ROOT_COL = "ticker"  # ORATS column that contains the underlying root
VERBOSE = True


def main() -> None:
    print(f"Raw ORATS root:          {RAW_ORATS_ROOT}")
    print(f"Output (by-ticker) root: {OUT_ROOT}")
    print(f"Tickers:                 {TICKERS}")
    if YEAR_WHITELIST is not None:
        print(f"Years:                   {sorted(str(y) for y in YEAR_WHITELIST)}")
    else:
        print("Years:                   ALL")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    extract_tickers_from_orats(
        raw_root=RAW_ORATS_ROOT,
        out_root=OUT_ROOT,
        tickers=TICKERS,
        year_whitelist=YEAR_WHITELIST,
        root_col=ROOT_COL,
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()