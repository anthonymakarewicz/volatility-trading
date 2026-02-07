#!/usr/bin/env python
from __future__ import annotations

import logging

from volatility_trading.utils.logging_config import setup_logging
from volatility_trading.config.paths import PROC_ORATS_DAILY_FEATURES
from volatility_trading.etl.orats.qc.daily_features.runner import (
    run_daily_features_qc,
)


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

PROC_ROOT = PROC_ORATS_DAILY_FEATURES
TICKER = "SPY"

# Logging
LOG_LEVEL = "INFO"  # "DEBUG" for more verbosity
LOG_COLORED = True
LOG_FILE = None  # e.g. "logs/qc_orats_daily_features.log"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    result = run_daily_features_qc(
        ticker=TICKER,
        proc_root=PROC_ROOT,
    )


if __name__ == "__main__":
    main()
