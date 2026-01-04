#!/usr/bin/env python
"""
Download ORATS API endpoint data to disk (raw).

Typical usage:
    python scripts/download_orats_api.py

Notes:
- Reads ORATS token from env var ORATS_API_KEY by default.
- Uses exchange_calendars (XNYS) inside the downloader to avoid holidays/weekends.
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

from volatility_trading.utils import setup_logging
from volatility_trading.data.orats_api_endpoints import (
    DownloadStrategy,
    get_endpoint_spec,
)
from volatility_trading.data.orats_downloader_api import download
from volatility_trading.config.paths import RAW_ORATS_API


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

# Token: recommended via env var
ORATS_API_KEY_ENV = "ORATS_API_KEY"

# Where to store raw API outputs (adjust to your layout)
RAW_ORATS_ROOT = RAW_ORATS_API

# Supported endpoint name (must exist in ENDPOINTS mapping)
ENDPOINT = "cores"
# ENDPOINT = "monies_implied"
# ENDPOINT = "cores"
# ENDPOINT = "summaries"

TICKERS = [
    "SPX", 
    "NDX", 
    "VIX",
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "TSLA",
    "NVDA",
    "MSFT",
]


# Only used for BY_TRADE_DATE endpoints (e.g., monies_implied)
YEAR_WHITELIST = None

# Optional: request only specific fields (None means “all fields returned”)
FIELDS = None

# Downloader behavior
SLEEP_S = 0.10
OVERWRITE = False

# Raw snapshot compression ("gz" or "none")
COMPRESSION = "gz"

# Logging
LOG_LEVEL = "DEBUG"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"
LOG_FILE = None  # e.g. "logs/download_orats_api.log"
LOG_COLORED = True


def main():
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    load_dotenv()
    TOKEN = os.environ.get(ORATS_API_KEY_ENV)
    if not TOKEN:
        raise RuntimeError(
            f"Missing ORATS API token. Set env var {ORATS_API_KEY_ENV}."
        )

    spec = get_endpoint_spec(ENDPOINT)
    logger.info("RAW API root: %s", RAW_ORATS_ROOT)
    logger.info("Endpoint:     %s", ENDPOINT)
    logger.info("Strategy:     %s", spec.strategy.value)
    logger.info("Tickers:      %s", TICKERS)
    logger.info("Years:        %s", YEAR_WHITELIST)
    logger.info("Fields:       %s", "ALL" if FIELDS is None else len(FIELDS))
    logger.info("Compression:  %s", COMPRESSION)

    RAW_ORATS_ROOT.mkdir(parents=True, exist_ok=True)

    year_whitelist = YEAR_WHITELIST
    if spec.strategy == DownloadStrategy.FULL_HISTORY:
        if YEAR_WHITELIST is not None:
            logger.warning(
                "YEAR_WHITELIST is ignored for FULL_HISTORY endpoint=%s",
                ENDPOINT,
            )
        year_whitelist = None

    result = download(
        token=TOKEN,
        endpoint=ENDPOINT,
        raw_root=RAW_ORATS_ROOT,
        tickers=TICKERS,
        year_whitelist=year_whitelist,
        fields=FIELDS,
        sleep_s=SLEEP_S,
        overwrite=OVERWRITE,
        compression=COMPRESSION,
    )

    return result


if __name__ == "__main__":
    _result = main()