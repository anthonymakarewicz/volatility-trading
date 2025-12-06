#!/usr/bin/env python
"""
Download raw ORATS SMV Strikes ZIP files from HostedFTP into data/raw/orats.

Expected remote layout (on HostedFTP):
    smvstrikes_2007_2012/2007/*.zip
    smvstrikes_2007_2012/2008/*.zip
    ...
    smvstrikes/2013/*.zip
    ...

Local layout after running:
    data/raw/orats/smvstrikes_2007_2012/2007/*.zip
    data/raw/orats/smvstrikes/2013/*.zip
    ...

Usage:
    1. Create a .env file (or export env vars) with:
        ORATS_FTP_USER=your_username
        ORATS_FTP_PASS=your_password

    2. Run:
        python scripts/download_orats_raw.py
"""
from pathlib import Path
import os

from dotenv import load_dotenv
from volatility_trading.data import download_orats_raw


# ---- CONFIG ----

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ORATS_ROOT = DATA_ROOT / "raw" / "orats"

# FTP host (both of these are used by HostedFTP for ORATS)
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


def main() -> None:
    load_dotenv()

    user = os.getenv("ORATS_FTP_USER")
    password = os.getenv("ORATS_FTP_PASS")

    if not user or not password:
        raise SystemExit(
            "Missing ORATS_FTP_USER or ORATS_FTP_PASS.\n"
            "Set them in your environment or in a .env file at the project root."
        )

    print(f"Raw ORATS root: {RAW_ORATS_ROOT}")
    if YEAR_WHITELIST is not None:
        print(f"Year whitelist: {sorted(str(y) for y in YEAR_WHITELIST)}")
    else:
        print("Year whitelist: ALL years in the remote base dirs.")

    download_orats_raw(
        host=HOST,
        user=user,
        password=password,
        remote_base_dirs=REMOTE_BASE_DIRS,
        raw_root=RAW_ORATS_ROOT,
        year_whitelist=YEAR_WHITELIST,
        validate_zip=VALIDATE_ZIP,
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()