from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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
class QCConfig:
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
class QCCheckResult:
    """Standard output of one QC check."""

    name: str
    severity: Severity
    grade: Grade
    passed: bool

    n_rows: int | None = None  # number of rows in df used for the check
    n_units: int | None = (
        None  # number of "units" for dataset checks (days, day/expiry)
    )
    n_viol: int | None = None
    viol_rate: float | None = None

    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QCRunResult:
    """Result of one QC run (one dataset, one ticker, one config)."""

    # What did we run?
    config: QCConfig
    ticker: str
    proc_root: Path
    checks: list[QCCheckResult]
    passed: bool
    parquet_path: Path | None = None

    # Results
    n_checks: int | None = None

    # Basic run stats
    duration_s: float | None = None
    n_rows: int | None = None
    n_rows_roi: int | None = None

    # Quick summary
    n_hard_fail: int | None = None
    n_soft_fail: int | None = None
    n_soft_warn: int | None = None

    # Artifacts
    out_summary_json: Path | None = None
    out_config_json: Path | None = None
