#!/usr/bin/env python
"""Download ORATS HostedFTP SMV Strikes ZIP files to disk (raw).

Typical usage:
    python scripts/download_orats_ftp.py

Notes
-----
- Reads FTP credentials from env vars (or a .env file at project root).
- The remote host and base directories are owned by the FTP downloader module
  and should usually not be changed from the script.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from volatility_trading.config.paths import RAW_ORATS_FTP
from volatility_trading.etl.orats.ftp import download
from volatility_trading.utils import setup_logging


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

# Credentials: recommended via env vars
ORATS_FTP_USER_ENV = "ORATS_FTP_USER"
ORATS_FTP_PASS_ENV = "ORATS_FTP_PASS"

# Where to store raw FTP outputs
RAW_ORATS_ROOT = RAW_ORATS_FTP

# Limit to specific years (as ints or strings) if you want to test first.
# Example: YEAR_WHITELIST = {2013, 2014}
YEAR_WHITELIST = None

# Download options
VALIDATE_ZIP = True
MAX_WORKERS = 3  # 1 for sequential, or 2–4 for some parallelism

# Logging
LOG_LEVEL = "INFO"
LOG_FMT_CONSOLE = "%(asctime)s %(levelname)s %(shortname)s - %(message)s"
LOG_FILE = None  # e.g. "logs/download_orats_ftp.log"
LOG_COLORED = True


def main() -> None:
    setup_logging(
        LOG_LEVEL,
        fmt_console=LOG_FMT_CONSOLE,
        log_file=LOG_FILE,
        colored=LOG_COLORED,
    )
    logger = logging.getLogger(__name__)

    load_dotenv()
    user = os.getenv(ORATS_FTP_USER_ENV)
    password = os.getenv(ORATS_FTP_PASS_ENV)
    if not user or not password:
        raise RuntimeError(
            "Missing ORATS FTP credentials. Set env vars "
            f"{ORATS_FTP_USER_ENV} and {ORATS_FTP_PASS_ENV}."
        )

    logger.info("RAW FTP root:  %s", RAW_ORATS_ROOT)
    logger.info(
        "Years:         %s",
        "ALL" if YEAR_WHITELIST is None else sorted(str(y) for y in YEAR_WHITELIST),
    )
    logger.info("Validate ZIP:  %s", VALIDATE_ZIP)
    logger.info("Max workers:   %s", MAX_WORKERS)

    RAW_ORATS_ROOT.mkdir(parents=True, exist_ok=True)

    # Host and remote base dirs are typically defaults owned by the downloader.
    result = download(
        user=user,
        password=password,
        raw_root=RAW_ORATS_ROOT,
        year_whitelist=YEAR_WHITELIST,
        validate_zip=VALIDATE_ZIP,
        max_workers=MAX_WORKERS,
    )


if __name__ == "__main__":
    main()