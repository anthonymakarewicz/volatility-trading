from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# TODO: Add a QCConfig dataclass
"""
@dataclass(frozen=True)
class OptionsChainQCConfig:
    ticker: str
    run_global: bool = True
    run_roi: bool = True

    # ROI definition
    roi_dte_min: int = 10
    roi_dte_max: int = 60
    roi_delta_min: float = 0.1
    roi_delta_max: float = 0.9

    # Bucketing used for soft-check reporting
    dte_bins: tuple[int, ...] = (0, 10, 30, 60, 180)
    delta_bins: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

    # Soft-check grading thresholds
    soft_thresholds: dict[str, float] = field(
        default_factory=lambda: {"mild": 0.01, "warn": 0.03, "fail": 0.10}
    )

    top_k_buckets: int = 10

@dataclass(frozen=True)
class QCRunResult:
    config: OptionsChainQCConfig
    checks: list[QCCheckResult]
    passed: bool
    out_json: Path | None = None
"""

class Severity(str, Enum):
    HARD = "HARD"  # structural constraints (should be ~0 violations)
    SOFT = "SOFT"  # arbitrage-ish / surface consistency checks
    INFO = "INFO"  # descriptive metrics, not pass/fail


class Grade(str, Enum):
    OK = "OK"
    MILD = "MILD"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class QCCheckResult:
    """Standard output of one QC check."""
    name: str
    severity: Severity
    grade: Grade
    passed: bool

    n_rows: int | None = None
    n_viol: int | None = None
    viol_rate: float | None = None  # n_viol / n_rows

    details: dict[str, Any] = field(default_factory=dict)