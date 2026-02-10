"""Instrument metadata used across ETL, QC, and downstream analytics.

Defines canonical mappings for:
- preferred OPRA roots used when multiple roots exist for an underlying
- option exercise style by symbol/root

Assumptions:
- keys are uppercase vendor symbols (underlyings or OPRA roots)
- values are stable configuration defaults, not runtime-discovered values
"""

from __future__ import annotations

from typing import Final, Literal

ExerciseStyle = Literal["AM", "EU"]

PREFERRED_OPRA_ROOT: Final[dict[str, str]] = {
    "SPX": "SPXW",  # For SPX, we want the PM-settled weeklies
}


OPTION_EXERCISE_STYLE: Final[dict[str, ExerciseStyle]] = {
    "SPX": "EU",
    "SPXW": "EU",
    "SPY": "AM",
    "NDX": "EU",
    "RUT": "EU",
    "AAPL": "AM",
}
