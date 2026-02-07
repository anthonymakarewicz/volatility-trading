from __future__ import annotations

from pathlib import Path

import polars as pl

from .hard.spec_types import HardSpec
from .hard.suite import run_hard_suite
from .info.spec_types import InfoSpec
from .info.suite import run_info_suite
from .reporting import write_config_json, write_summary_json
from .soft.spec_types import SoftSpec
from .soft.suite import run_soft_suite
from .types import (
    Grade,
    QCCheckResult,
    QCConfig,
    Severity
)


def write_json_reports(
    *,
    write_json: bool,
    out_json: Path | str | None,
    parquet_path: Path | None,
    results: list[QCCheckResult],
    config: QCConfig,
    missing_error_prefix: str | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Optionally write qc_summary.json and qc_config.json."""
    if not write_json:
        return parquet_path, None, None

    if out_json is None:
        if parquet_path is None:
            raise FileNotFoundError(
                "parquet_path is required when out_json is None"
            )

        if not parquet_path.exists():
            prefix = missing_error_prefix or "Processed dataset not found"
            raise FileNotFoundError(f"{prefix}: {parquet_path}")

        out_summary_json = parquet_path.parent / "qc_summary.json"
    else:
        out_summary_json = Path(out_json)

    out_config_json = out_summary_json.with_name("qc_config.json")

    write_summary_json(out_summary_json, results)
    write_config_json(out_config_json, config)

    return parquet_path, out_summary_json, out_config_json


def compute_outcome(results: list[QCCheckResult]) -> tuple[bool, int, int, int]:
    """Compute overall pass/fail + a few run-level counts."""
    passed = all(
        r.passed
        for r in results
        if r.severity in {Severity.HARD, Severity.SOFT}
    )

    n_hard_fail = sum(
        1
        for r in results
        if (r.severity == Severity.HARD and not r.passed)
    )
    n_soft_fail = sum(
        1
        for r in results
        if (r.severity == Severity.SOFT and r.grade == Grade.FAIL)
    )
    n_soft_warn = sum(
        1
        for r in results
        if (r.severity == Severity.SOFT and r.grade == Grade.WARN)
    )

    return passed, n_hard_fail, n_soft_fail, n_soft_warn


def run_all_checks(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    config: QCConfig,
    exercise_style: str | None,
    hard_specs: list[HardSpec],
    soft_specs: list[SoftSpec],
    info_specs: list[InfoSpec] | None = None,
) -> list[QCCheckResult]:
    """Run HARD, SOFT, INFO suites and return results."""
    results: list[QCCheckResult] = []
    results.extend(
        run_hard_suite(
            df_global=df_global,
            hard_specs=hard_specs,
        )
    )
    results.extend(
        run_soft_suite(
            df_global=df_global,
            df_roi=df_roi,
            config=config,
            exercise_style=exercise_style,
            soft_specs=soft_specs,
        )
    )
    if info_specs is not None:
        results.extend(
            run_info_suite(
                df_global=df_global,
                df_roi=df_roi,
                info_specs=info_specs,
            )
        )
    return results
