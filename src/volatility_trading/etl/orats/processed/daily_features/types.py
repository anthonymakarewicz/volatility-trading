"""Dataclasses for processed ORATS daily-features build outputs and stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuildStats:
    """Internal mutable counters used during build (populated only if enabled)."""

    n_rows_input_total: int | None = None

    n_rows_input_by_endpoint: dict[str, int] = field(default_factory=dict)
    n_rows_after_dedupe_by_endpoint: dict[str, int] = field(default_factory=dict)

    n_rows_spine: int | None = None
    n_rows_written: int | None = None


@dataclass(frozen=True)
class BuildDailyFeaturesResult:
    """Summary of a processed daily-features build for one ticker."""

    ticker: str
    out_path: Path
    duration_s: float

    n_rows_written: int
    n_rows_input_total: int | None = None

    n_rows_spine: int | None = None

    n_rows_input_by_endpoint: dict[str, int] = field(default_factory=dict)
    n_rows_after_dedupe_by_endpoint: dict[str, int] = field(default_factory=dict)
