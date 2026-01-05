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
import logging

from volatility_trading.config.paths import RAW_ORATS_FTP, INTER_ORATS_FTP
from volatility_trading.data import extract_tickers_from_orats
from volatility_trading.utils.logging_config import setup_logging


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

RAW_ORATS_ROOT = RAW_ORATS_FTP
OUT_ROOT = INTER_ORATS_FTP

# Choose which tickers you want to extract
TICKERS = [
    "SPX",
]

# Restrict to specific years if you want (ints or strings), or None for all:
# YEAR_WHITELIST = {2013, 2014}
YEAR_WHITELIST = [2007]

# Logging
LOG_LEVEL = "DEBUG"  # change to "DEBUG" for more detail
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s" 
LOG_FILE = None  # e.g. "logs/extract_orats_tickers.log"
LOG_COLORED = True


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    logger.info("Raw ORATS root:          %s", RAW_ORATS_ROOT)
    logger.info("Output (by-ticker) root: %s", OUT_ROOT)
    logger.info("Tickers:                 %s", TICKERS)
    if YEAR_WHITELIST is not None:
        logger.info("Years:                   %s", sorted(str(y) for y in YEAR_WHITELIST))
    else:
        logger.info("Years:                   ALL")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    extract_tickers_from_orats(
        raw_root=RAW_ORATS_ROOT,
        out_root=OUT_ROOT,
        tickers=TICKERS,
        year_whitelist=YEAR_WHITELIST,
    )


if __name__ == "__main__":
    main()