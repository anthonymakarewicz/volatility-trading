"""
Internal helpers for the ORATS daily-features QC runner.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from volatility_trading.datasets import daily_features_path

from ..reporting import write_config_json, write_summary_json
from ..runners import run_hard_check, run_soft_check
from ..soft.summarizers import summarize_by_bucket
from ..types import (
    Grade,
    QCCheckResult,
    QCConfig,
    Severity
)
from .specs import get_hard_specs, get_soft_specs

logger = logging.getLogger(__name__)


def run_all_checks(
    *,
    df_global: pl.DataFrame,
    config: QCConfig,
) -> list[QCCheckResult]:
    """Run HARD and SOFT suites for daily-features QC."""
    results: list[QCCheckResult] = []

    for spec in get_hard_specs():
        results.append(
            run_hard_check(
                name=spec.name,
                df=df_global,
                predicate_expr=spec.predicate_expr,
                sample_n=spec.sample_n,
                sample_cols=spec.sample_cols,
            )
        )

    soft_thresholds = dict(config.soft_thresholds)

    for spec in get_soft_specs():
        summarizer = summarize_by_bucket if spec.summarize_by_bucket else None
        summarizer_kwargs = (
            {"dte_bins": config.dte_bins, "delta_bins": config.delta_bins}
            if summarizer is not None
            else {}
        )

        if spec.by_option_type:
            for opt in ["C", "P"]:
                results.append(
                    run_soft_check(
                        name=f"GLOBAL_{spec.base_name}_{opt}",
                        df=df_global,
                        flagger=spec.flagger,
                        violation_col=spec.violation_col,
                        flagger_kwargs={
                            "option_type": opt,
                            **dict(spec.flagger_kwargs),
                        },
                        summarizer=summarizer,
                        summarizer_kwargs=dict(summarizer_kwargs),
                        thresholds=spec.thresholds or soft_thresholds,
                        top_k_buckets=config.top_k_buckets,
                        sample_cols=spec.sample_cols,
                        sample_n=5,
                    )
                )
        else:
            results.append(
                run_soft_check(
                    name=f"GLOBAL_{spec.base_name}",
                    df=df_global,
                    flagger=spec.flagger,
                    violation_col=spec.violation_col,
                    flagger_kwargs=dict(spec.flagger_kwargs),
                    summarizer=summarizer,
                    summarizer_kwargs=dict(summarizer_kwargs),
                    thresholds=spec.thresholds or soft_thresholds,
                    top_k_buckets=config.top_k_buckets,
                    sample_cols=spec.sample_cols,
                    sample_n=5,
                )
            )

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
            parquet_path = daily_features_path(proc_root, ticker)

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Processed daily features not found: {parquet_path}"
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
