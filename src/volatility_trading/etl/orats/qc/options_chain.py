"""
volatility_trading.etl.orats.qc.options_chain

QC runner for the processed ORATS *options chain* dataset.

This module is **read-only**: it does not drop or modify rows. It reports:
- **HARD checks**: must-pass invariants (null keys, trade_date <= expiry_date,
  bid/ask sanity: non-negative, not crossed).
- **SOFT checks**: diagnostics (strike/maturity monotonicity, locked/one-sided
  quotes, wide spreads).
- **INFO checks**: informal diagnostics only (do not fail).
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .checks_info import (
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics,
)
from .report import log_check, write_config_json, write_summary_json
from .runners import run_info_check
from .hard.suite import run_hard_suite
from .soft.suite import run_soft_suite
from .types import Grade, QCCheckResult, QCConfig, QCRunResult, Severity
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    options_chain_path,
    scan_options_chain,
)


logger = logging.getLogger(__name__)


def _read_exercise_style(
    *,
    parquet_path: Path | None,
) -> str | None:
    """Read exercise_style from manifest.json next to the parquet file."""
    if parquet_path is None:
        return None

    try:
        manifest_path = parquet_path.parent / "manifest.json"
        if not manifest_path.exists():
            return None

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        style = payload.get("exercise_style", None)
        if style not in {"EU", "AM"}:
            return None
        return style

    except Exception:
        # Don't fail QC because of manifest parsing issues
        return None


def _apply_roi_filter(
    df: pl.DataFrame,
    *,
    dte_min: int = 10,
    dte_max: int = 60,
    delta_min: float = 0.1,
    delta_max: float = 0.9,
) -> pl.DataFrame:
    """Filter to the region-of-interest (ROI) used by soft-check summaries."""
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )


def _run_info_checks(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
) -> list[QCCheckResult]:
    """Run informational checks (always pass) and store metrics in details."""
    results: list[QCCheckResult] = []

    for label, dfx in [("GLOBAL", df_global), ("ROI", df_roi)]:
        results.append(
            run_info_check(
                name=f"{label}_volume_oi_metrics",
                df=dfx,
                summarizer=summarize_volume_oi_metrics,
            )
        )
        results.append(
            run_info_check(
                name=f"{label}_risk_free_rate_metrics",
                df=dfx,
                summarizer=summarize_risk_free_rate_metrics,
            )
        )

    return results


def run_qc(
    *,
    ticker: str,
    proc_root: Path | str,
    out_json: Path | str | None = None,
    write_json: bool = True,
    dte_bins: Sequence[int | float] = (0, 10, 30, 60, 180),
    delta_bins: Sequence[float] = (0.0, 0.05, 0.1, 0.3, 0.7, 0.9, 0.95, 1.0),
    roi_dte_min: int = 10,
    roi_dte_max: int = 60,
    roi_delta_min: float = 0.1,
    roi_delta_max: float = 0.9,
    top_k_buckets: int = 10,
) -> QCRunResult:
    """Run QC on the processed options chain for one underlying."""
    t0 = time.perf_counter()

    proc_root_p = Path(proc_root)
    ticker_s = str(ticker).strip().upper()

    parquet_path: Path | None
    try:
        parquet_path = options_chain_path(proc_root_p, ticker_s)
    except Exception:
        parquet_path = None

    # Determine exercise style (EU/AM) from manifest.json next to parquet.
    exercise_style = _read_exercise_style(
        parquet_path=parquet_path
    )

    if exercise_style is None:
        logger.warning(
            "exercise_style missing/invalid in manifest.json -> "
            "skip EU/AM specific soft checks (ticker=%s).",
            ticker_s,
        )

    config = QCConfig(
        ticker=ticker_s,
        roi_dte_min=roi_dte_min,
        roi_dte_max=roi_dte_max,
        roi_delta_min=roi_delta_min,
        roi_delta_max=roi_delta_max,
        dte_bins=tuple(dte_bins),
        delta_bins=tuple(delta_bins),
        top_k_buckets=top_k_buckets,
    )

    # Load processed options chain as wide format
    lf = scan_options_chain(ticker_s, proc_root=proc_root_p)
    df = options_chain_wide_to_long(lf).collect()

    n_rows: int | None = int(df.height)

    logger.info(
        "QC start ticker=%s rows=%s roi=(dte=%s..%s |delta|=%s..%s) parquet=%s",
        ticker_s,
        n_rows,
        config.roi_dte_min,
        config.roi_dte_max,
        config.roi_delta_min,
        config.roi_delta_max,
        parquet_path,
    )

    df_roi = _apply_roi_filter(
        df,
        dte_min=config.roi_dte_min,
        dte_max=config.roi_dte_max,
        delta_min=config.roi_delta_min,
        delta_max=config.roi_delta_max,
    )
    n_rows_roi: int | None = int(df_roi.height)

    # Run checks
    results: list[QCCheckResult] = []
    results.extend(run_hard_suite(df_global=df))
    results.extend(
        run_soft_suite(
            df_global=df,
            df_roi=df_roi,
            config=config,
            exercise_style=exercise_style,
        )
    )
    results.extend(_run_info_checks(df_global=df, df_roi=df_roi))

    for r in results:
        log_check(logger, r)

    # Write JSON reports
    out_summary_json: Path | None = None
    out_config_json: Path | None = None

    if write_json:
        if out_json is None:
            if parquet_path is None:
                parquet_path = options_chain_path(proc_root_p, ticker_s)
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

    # PASS/FAIL:
    # - HARD must all pass
    # - SOFT fails are FAIL grade only
    passed = all(
        r.passed for r in results
        if r.severity in {Severity.HARD, Severity.SOFT}
    )

    n_hard_fail = sum(
        1 for r in results
        if (r.severity == Severity.HARD and not r.passed)
    )
    n_soft_fail = sum(
        1 for r in results
        if (r.severity == Severity.SOFT and r.grade == Grade.FAIL)
    )
    n_soft_warn = sum(
        1 for r in results
        if (r.severity == Severity.SOFT and r.grade == Grade.WARN)
    )

    duration_s = time.perf_counter() - t0

    return QCRunResult(
        config=config,
        ticker=ticker_s,
        proc_root=proc_root_p,
        parquet_path=parquet_path,
        checks=results,
        passed=passed,
        n_checks=len(results),
        duration_s=duration_s,
        n_rows=n_rows,
        n_rows_roi=n_rows_roi,
        n_hard_fail=n_hard_fail,
        n_soft_fail=n_soft_fail,
        n_soft_warn=n_soft_warn,
        out_summary_json=out_summary_json,
        out_config_json=out_config_json,
    )