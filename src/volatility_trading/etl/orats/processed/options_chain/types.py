"""volatility_trading.etl.orats.processed.options_chain.types

Public and internal dataclasses for the ORATS processed options-chain builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildOptionsChainResult:
    """Summary of a processed options-chain build for one ticker.

    Notes
    -----
    When `collect_stats=False`, most counters are left as None to avoid
    additional scans.
    """

    ticker: str
    out_path: Path
    duration_s: float

    # Final materialised output
    n_rows_written: int

    # Optional stats (only when collect_stats=True)
    n_rows_input: int | None
    n_rows_after_dedupe: int | None

    n_rows_yield_input: int | None
    n_rows_yield_after_dedupe: int | None
    n_rows_join_missing_yield: int | None

    n_rows_after_trading: int | None
    n_rows_after_hard: int | None


@dataclass
class BuildStats:
    """Internal mutable counters used during build (populated only if enabled)."""

    n_rows_input: int | None = None
    n_rows_after_dedupe: int | None = None

    n_rows_yield_input: int | None = None
    n_rows_yield_after_dedupe: int | None = None
    n_rows_join_missing_yield: int | None = None

    n_rows_after_trading: int | None = None
    n_rows_after_hard: int | None = None
