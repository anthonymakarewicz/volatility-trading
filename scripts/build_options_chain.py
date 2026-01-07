#!/usr/bin/env python
"""
Build cleaned ORATS panels for a list of tickers from intermediate by-ticker
parquet files.

This is a thin wrapper around
    volatility_trading.etl.orats.processed.build_options_chain
so you can regenerate panels from the command line.
"""

import logging

from volatility_trading.utils import setup_logging
from volatility_trading.etl.orats.processed import build_options_chain
from volatility_trading.config.paths import (
    INTER_ORATS_FTP,
    PROC_ORATS_OPTIONS_CHAIN,
)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

PROC_ROOT = PROC_ORATS_OPTIONS_CHAIN
INTER_ROOT = INTER_ORATS_FTP

# Underlyings to build panels for
TICKERS = ["SPY"]  # e.g. ["SPX", "SPY", "QQQ", "IWM", ...]

# Restrict to a subset of years, or None for all available
YEARS = None # e.g. range(2007, 2026)

# DTE and moneyness bands used in the cleaning step
DTE_MIN = 1
DTE_MAX = 252
MONEYNESS_MIN = 0.5
MONEYNESS_MAX = 1.5

# Optional column override; if None, uses CORE_ORATS_WIDE_COLUMNS
COLUMNS = None

# Logging
LOG_LEVEL = "INFO"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"
LOG_FILE = None  # e.g. "logs/build_options_chain.log"
LOG_COLORED = True


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    logger.info("INTER root: %s", INTER_ROOT)
    logger.info("PROC root:  %s", PROC_ROOT)
    logger.info("Tickers:    %s", TICKERS)
    logger.info("Years:      %s", YEARS)
    logger.info("DTE:        [%s, %s]", DTE_MIN, DTE_MAX)
    logger.info("Moneyness:  [%s, %s]", MONEYNESS_MIN, MONEYNESS_MAX)
    logger.info("Columns:    %s", "DEFAULT" if COLUMNS is None else len(COLUMNS))

    for ticker in TICKERS:
        logger.info("Building options chain ticker=%s", ticker)
        build_options_chain(
            inter_root=INTER_ROOT,
            proc_root=PROC_ROOT,
            ticker=ticker,
            years=YEARS,
            dte_min=DTE_MIN,
            dte_max=DTE_MAX,
            moneyness_min=MONEYNESS_MIN,
            moneyness_max=MONEYNESS_MAX,
            columns=COLUMNS,
        )


if __name__ == "__main__":
    main()