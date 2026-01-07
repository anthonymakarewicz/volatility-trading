#!/usr/bin/env python
"""
Build cleaned ORATS panels for a list of tickers from intermediate by-ticker
parquet files.

This is a thin wrapper around
    volatility_trading.etl.orats.processed.build_options_chain
so you can regenerate panels from the command line.
"""

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


def main() -> None:
    for ticker in TICKERS:
        print(f"\n=== Building panel for {ticker} ===")
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
            verbose=True,
        )


if __name__ == "__main__":
    main()