#!/usr/bin/env python
"""
Build processed options-chain panels from intermediate data.

This script is a thin wrapper around:
    volatility_trading.etl.orats.processed.build_options_chain

It builds a per-ticker options chain under the processed root and (optionally)
merges dividend/yield information from the ORATS API `monies_implied` endpoint
(intermediate parquet) to populate/overwrite `dividend_yield`.
"""

import logging

from volatility_trading.utils import setup_logging
from volatility_trading.etl.orats.processed import build_options_chain
from volatility_trading.config.paths import (
    INTER_ORATS_API,
    INTER_ORATS_FTP,
    PROC_ORATS_OPTIONS_CHAIN,
)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

PROC_ROOT = PROC_ORATS_OPTIONS_CHAIN
INTER_STRIKES_ROOT = INTER_ORATS_FTP
INTER_API_ROOT = INTER_ORATS_API
MONIES_IMPLIED_ENDPOINT = "monies_implied"

# Collect and log build stats (dedupe drops, join misses, filter drops)
COLLECT_STATS = True

# Underlyings to build panels for
TICKERS = ["SPX"]  # e.g. ["SPX", "SPY", "QQQ", "IWM", ...]

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

    logger.info("INTER strikes root: %s", INTER_STRIKES_ROOT)
    logger.info("INTER API root:     %s", INTER_API_ROOT)
    logger.info("API yield endpoint: %s", MONIES_IMPLIED_ENDPOINT)
    logger.info("PROC root:  %s", PROC_ROOT)
    logger.info("Tickers:    %s", TICKERS)
    logger.info("Years:      %s", YEARS)
    logger.info("DTE:        [%s, %s]", DTE_MIN, DTE_MAX)
    logger.info("Moneyness:  [%s, %s]", MONEYNESS_MIN, MONEYNESS_MAX)
    logger.info("Columns:    %s", "DEFAULT" if COLUMNS is None else len(COLUMNS))
    logger.info("Collect stats: %s", COLLECT_STATS)

    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        logger.info("Building options chain ticker=%s", ticker)
        result = build_options_chain(
            inter_root=INTER_STRIKES_ROOT,
            proc_root=PROC_ROOT,
            ticker=ticker,
            years=YEARS,
            dte_min=DTE_MIN,
            dte_max=DTE_MAX,
            moneyness_min=MONEYNESS_MIN,
            moneyness_max=MONEYNESS_MAX,
            columns=COLUMNS,
            collect_stats=COLLECT_STATS,
        )


if __name__ == "__main__":
    main()