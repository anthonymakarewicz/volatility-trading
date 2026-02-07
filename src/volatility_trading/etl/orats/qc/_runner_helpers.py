"""
Internal helpers for the ORATS QC runner.

This module holds small orchestration utilities used by `runner.py`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from volatility_trading.datasets import (
    options_chain_path,
    options_chain_wide_to_long,
    scan_options_chain,
)

from .hard.suite import run_hard_suite
from .info.suite import run_info_suite
from .reporting import write_config_json, write_summary_json
from .soft.suite import run_soft_suite
from .types import Grade, QCCheckResult, QCConfig, Severity

logger = logging.getLogger(__name__)


def get_parquet_path(proc_root: Path, ticker: str) -> Path | None:
    """Best-effort locate the processed options-chain parquet."""
    try:
        return options_chain_path(proc_root, ticker)
    except Exception:
        return None


def read_exercise_style(*, parquet_path: Path | None) -> str | None:
    """Best-effort read exercise_style (EU/AM) from manifest.json."""
    if parquet_path is None:
        return None

    try:
        manifest_path = parquet_path.parent / "manifest.json"
        if not manifest_path.exists():
            return None

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        params = payload.get("params")
        if not isinstance(params, dict):
            return None

        style = params.get("exercise_style", None)
        return style if style in {"EU", "AM"} else None
    except Exception:
        return None


def load_options_chain_df(*, ticker: str, proc_root: Path) -> pl.DataFrame:
    """Load processed options chain as a long Polars DataFrame."""
    lf = scan_options_chain(ticker, proc_root=proc_root)
    return options_chain_wide_to_long(lf).collect()


def apply_roi_filter(
    df: pl.DataFrame,
    *,
    dte_min: int = 10,
    dte_max: int = 60,
    delta_min: float = 0.1,
    delta_max: float = 0.9,
) -> pl.DataFrame:
    """Filter to ROI used by soft/info reporting."""
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )


def run_all_checks(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    config: QCConfig,
    exercise_style: str | None,
) -> list[QCCheckResult]:
    """Run HARD, SOFT, INFO suites and return results."""
    results: list[QCCheckResult] = []
    results.extend(run_hard_suite(df_global=df_global))
    results.extend(
        run_soft_suite(
            df_global=df_global,
            df_roi=df_roi,
            config=config,
            exercise_style=exercise_style,
        )
    )
    results.extend(run_info_suite(df_global=df_global, df_roi=df_roi))
    return results


def write_json_reports(
    *,
    write_json: bool,
    out_json: Path | str | None,
    parquet_path: Path | None,
    proc_root: Path,
    ticker: str,
    results: list[QCCheckResult],
    config: QCConfig,
) -> tuple[Path | None, Path | None, Path | None]:
    """Optionally write qc_summary.json and qc_config.json."""
    if not write_json:
        return parquet_path, None, None

    if out_json is None:
        if parquet_path is None:
            parquet_path = options_chain_path(proc_root, ticker)

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Processed options chain not found: {parquet_path}"
            )

        out_summary_json = parquet_path.parent / "qc_summary.json"
    else:
        out_summary_json = Path(out_json)

    out_config_json = out_summary_json.with_name("qc_config.json")

    write_summary_json(out_summary_json, results)
    write_config_json(out_config_json, config)

    logger.info("QC summary written: %s", out_summary_json)
    logger.info("QC config written:  %s", out_config_json)

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