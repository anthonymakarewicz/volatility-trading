#!/usr/bin/env python
"""
Build processed daily-features panels from intermediate ORATS API data.

This script is a thin wrapper around:
    volatility_trading.etl.orats.processed.daily_features.build
"""

import logging

from volatility_trading.config.paths import (
    INTER_ORATS_API,
    PROC_ORATS_DAILY_FEATURES
)
from volatility_trading.utils import setup_logging
from volatility_trading.etl.orats.processed.daily_features import build


PROC_ROOT = PROC_ORATS_DAILY_FEATURES
INTER_API_ROOT = INTER_ORATS_API

COLLECT_STATS = True
TICKERS = ["AAPL"]

ENDPOINTS = ("summaries", "hvs")
BASE_ENDPOINT = "summaries"

LOG_LEVEL = "INFO"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"
LOG_FILE = None
LOG_COLORED = True


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    logger.info("INTER API root: %s", INTER_API_ROOT)
    logger.info("PROC root:      %s", PROC_ROOT)
    logger.info("Tickers:        %s", TICKERS)
    logger.info("Endpoints:      %s", ENDPOINTS)
    logger.info("Base endpoint:  %s", BASE_ENDPOINT)

    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        logger.info("Building daily features ticker=%s", ticker)
        _result = build(
            inter_api_root=INTER_API_ROOT,
            proc_root=PROC_ROOT,
            ticker=ticker,
            endpoints=ENDPOINTS,
            collect_stats=COLLECT_STATS,
        )


if __name__ == "__main__":
    main()