#!/usr/bin/env python
from __future__ import annotations

import logging

from volatility_trading.utils import setup_logging
from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN
from volatility_trading.etl.orats.qc.options_chain import run_qc


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

PROC_ROOT = PROC_ORATS_OPTIONS_CHAIN
TICKER = "SPY"

# Logging
LOG_LEVEL = "INFO"  # "DEBUG" for more verbosity
LOG_COLORED = True
LOG_FILE = None  # e.g. "logs/extract_orats_api.log"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    run_qc(
        ticker=TICKER,
        proc_root=PROC_ROOT,
    )


if __name__ == "__main__":
    main()
