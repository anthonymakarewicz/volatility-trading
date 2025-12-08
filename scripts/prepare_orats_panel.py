#!/usr/bin/env python
"""
Build cleaned ORATS panels for a list of tickers from intermediate by-ticker
parquet files.
"""

from pathlib import Path

from volatility_trading.config.paths import INTER_ORATS_BY_TICKER, PROC_ORATS
from volatility_trading.data.orats_panel import build_orats_panel_for_ticker


TICKERS = ["SPX"]  # or ["SPX", "SPY", "QQQ", ...]
YEARS = None       # or range(2010, 2026)


def main() -> None:
    for ticker in TICKERS:
        build_orats_panel_for_ticker(
            ticker=ticker,
            inter_root=INTER_ORATS_BY_TICKER,
            proc_root=PROC_ORATS,
            years=YEARS,
            min_volume=1,
            max_bid_ask_spread=None,
            verbose=True,
        )


if __name__ == "__main__":
    main()