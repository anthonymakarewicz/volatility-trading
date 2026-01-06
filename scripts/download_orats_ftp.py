#!/usr/bin/env python
"""
Download raw ORATS SMV Strikes ZIP files from HostedFTP into data/raw/options/orats
(using the configured RAW_ORATS path).

Expected remote layout (on HostedFTP):
    smvstrikes_2007_2012/2007/*.zip
    smvstrikes_2007_2012/2008/*.zip
    ...
    smvstrikes/2013/*.zip
    ...

Local layout after running (example if RAW_ORATS points to data/raw/options/orats):
    data/raw/options/orats/smvstrikes_2007_2012/2007/*.zip
    data/raw/options/orats/smvstrikes/2013/*.zip
    ...

Usage:
    1. Create a .env file (or export env vars) with:
        ORATS_FTP_USER=your_orats_username
        ORATS_FTP_PASS=your_orats_password

    2. Run:
        python scripts/download_orats_raw.py
"""

import os
from dotenv import load_dotenv

from volatility_trading.config.paths import RAW_ORATS_FTP
from volatility_trading.etl.orats.ftp import download


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

ORATS_FTP_USER_ENV = "ORATS_FTP_USER"
ORATS_FTP_PASS_ENV = "ORATS_FTP_PASS"

HOST = "orats.hostedftp.com"  # or "de1.hostedftp.com"
REMOTE_BASE_DIRS = [
    "smvstrikes_2007_2012",  # 2007–2012
    "smvstrikes",            # 2013–present
]

# Limit to specific years (as ints or strings) if you want to test first.
# Example: YEAR_WHITELIST = {2013, 2014}
YEAR_WHITELIST = None

VALIDATE_ZIP = True
VERBOSE = True
MAX_WORKERS = 3  # or 1 for sequential, or 2–4 for some parallelism


def main() -> None:
    load_dotenv()
    user = os.getenv(ORATS_FTP_USER_ENV)
    password = os.getenv(ORATS_FTP_PASS_ENV)

    if not user or not password:
        raise SystemExit(
            "Missing ORATS_FTP_USER or ORATS_FTP_PASS.\n"
            "Set them in your environment or in a .env file at the project root."
        )

    print(f"Raw ORATS root: {RAW_ORATS_FTP}")
    if YEAR_WHITELIST is not None:
        print(f"Year whitelist: {sorted(str(y) for y in YEAR_WHITELIST)}")
    else:
        print("Year whitelist: ALL years in the remote base dirs.")

    download(
        host=HOST,
        user=user,
        password=password,
        remote_base_dirs=REMOTE_BASE_DIRS,
        raw_root=RAW_ORATS_FTP,
        year_whitelist=YEAR_WHITELIST,
        validate_zip=VALIDATE_ZIP,
        verbose=VERBOSE,
        max_workers=MAX_WORKERS,
    )


if __name__ == "__main__":
    main()