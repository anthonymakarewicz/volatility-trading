"""
volatility_trading.config.instruments

Configuration constants for instruments used in volatility trading.
"""
from __future__ import annotations


PREFERRED_OPRA_ROOT: dict[str, str] = {
    "SPX": "SPXW", # For SPX, we want the PM-settled weeklies
}


OPTION_EXERCISE_STYLE = {
    "SPX": "EU",
    "SPXW": "EU",
    "SPY": "AM",
    "NDX": "EU",
    "RUT": "EU",
    "SPY": "AM",
    "AAPL": "AM",
}