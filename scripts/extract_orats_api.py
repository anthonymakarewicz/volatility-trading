

"""Extract ORATS API raw snapshots into intermediate parquet.

This script converts raw ORATS API payloads stored under your raw data folder
(JSON or JSON.GZ, as produced by the API downloader) into ticker-centric
intermediate parquet files.

Typical usage
-------------
Run from the repo root:

    python scripts/extract_orats_api.py

Adjust the configuration constants in this file (ENDPOINT, TICKERS, YEARS,
paths, etc.) to match what you want to extract.
"""

from __future__ import annotations

import logging

from volatility_trading.utils.logging_config import setup_logging
from volatility_trading.etl.orats.api import extract
from volatility_trading.config.paths import RAW_ORATS_API, INTER_ORATS_API


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Data locations (edit to your paths)
RAW_ORATS_ROOT = RAW_ORATS_API
INTER_ORATS_ROOT = INTER_ORATS_API

# Endpoint to extract (must exist in orats_api_endpoints.ENDPOINTS)
ENDPOINT = "summaries"

# Extraction scope
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

# Only needed for BY_TRADE_DATE endpoints. Ignored for FULL_HISTORY endpoints.
YEAR_WHITELIST = list(range(2007, 2026))

# Raw snapshot compression mode used in the raw folder.
# Must match how you downloaded raw ("gz" or "none").
RAW_COMPRESSION = "gz"

# Logging
LOG_LEVEL = "INFO"  # "DEBUG" for more verbosity
LOG_COLORED = True
LOG_FILE = None  # e.g. "logs/extract_orats_api.log"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"

# Extraction options
OVERWRITE = True
PARQUET_COMPRESSION = "zstd"  # "zstd" is a good default for intermediate
USE_BOUNDS = True  # apply endpoint bounds (out-of-range -> null)


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    logger.info("RAW API root:         %s", RAW_ORATS_API)
    logger.info("Intermediate API root:%s", INTER_ORATS_API)
    logger.info("Endpoint:             %s", ENDPOINT)
    logger.info("Tickers:              %s", TICKERS)
    logger.info("Years:                %s", YEAR_WHITELIST)
    logger.info("Raw compression:      %s", RAW_COMPRESSION)
    logger.info("Overwrite:            %s", OVERWRITE)
    logger.info("Parquet compression:  %s", PARQUET_COMPRESSION)
    logger.info("Use bounds:           %s", USE_BOUNDS)

    RAW_ORATS_API.mkdir(parents=True, exist_ok=True)
    INTER_ORATS_API.mkdir(parents=True, exist_ok=True)

    result = extract(
        endpoint=ENDPOINT,
        raw_root=RAW_ORATS_ROOT,
        intermediate_root=INTER_ORATS_ROOT,
        tickers=TICKERS,
        year_whitelist=YEAR_WHITELIST,
        compression=RAW_COMPRESSION,
        overwrite=OVERWRITE,
        parquet_compression=PARQUET_COMPRESSION,
        use_bounds=USE_BOUNDS,
    )


if __name__ == "__main__":
    main()