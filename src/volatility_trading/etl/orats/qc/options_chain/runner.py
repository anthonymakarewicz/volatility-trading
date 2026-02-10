"""QC runner for processed ORATS options-chain panels.

Runs HARD, SOFT, and INFO suites and optionally writes JSON QC artifacts.
This module is read-only and does not mutate dataset rows.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from ..common_helpers import compute_outcome, run_all_checks, write_json_reports
from ..reporting import log_check
from ..types import QCConfig, QCRunResult
from .hard.specs import get_hard_specs
from .helpers import (
    apply_roi_filter,
    get_parquet_path,
    load_options_chain_df,
    read_exercise_style,
)
from .info.specs import get_info_specs
from .soft.specs import get_soft_specs

logger = logging.getLogger(__name__)


def run_options_chain_qc(
    *,
    ticker: str,
    proc_root: Path | str,
    out_json: Path | str | None = None,
    write_json: bool = True,
    dte_bins: Sequence[int] = (0, 10, 30, 60, 180),
    delta_bins: Sequence[float] = (0.0, 0.05, 0.1, 0.3, 0.7, 0.9, 0.95, 1.0),
    roi_dte_min: int = 10,
    roi_dte_max: int = 60,
    roi_delta_min: float = 0.1,
    roi_delta_max: float = 0.9,
    top_k_buckets: int = 10,
) -> QCRunResult:
    """Run QC on one processed options-chain ticker panel.

    Args:
        ticker: Underlying ticker symbol.
        proc_root: Root directory containing processed options-chain panels.
        out_json: Optional explicit path for `qc_summary.json`.
        write_json: Write `qc_summary.json` and `qc_config.json` when `True`.
        dte_bins: DTE bucket edges for SOFT diagnostics.
        delta_bins: Absolute-delta bucket edges for SOFT diagnostics.
        roi_dte_min: Minimum DTE for ROI subset checks.
        roi_dte_max: Maximum DTE for ROI subset checks.
        roi_delta_min: Minimum absolute delta for ROI subset checks.
        roi_delta_max: Maximum absolute delta for ROI subset checks.
        top_k_buckets: Maximum number of bucket summaries to keep per check.

    Returns:
        Run summary with check results, outcome counts, timing, and artifact paths.
    """
    t0 = time.perf_counter()

    proc_root_p = Path(proc_root)
    ticker_s = str(ticker).strip().upper()

    parquet_path = get_parquet_path(proc_root_p, ticker_s)

    exercise_style = read_exercise_style(parquet_path=parquet_path)
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

    df = load_options_chain_df(ticker=ticker_s, proc_root=proc_root_p)
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

    df_roi = apply_roi_filter(
        df,
        dte_min=config.roi_dte_min,
        dte_max=config.roi_dte_max,
        delta_min=config.roi_delta_min,
        delta_max=config.roi_delta_max,
    )
    n_rows_roi: int | None = int(df_roi.height)

    results = run_all_checks(
        df_global=df,
        df_roi=df_roi,
        config=config,
        exercise_style=exercise_style,
        hard_specs=get_hard_specs(),
        soft_specs=get_soft_specs(exercise_style=exercise_style),
        info_specs=get_info_specs(),
    )

    for r in results:
        log_check(logger, r)

    parquet_path, out_summary_json, out_config_json = write_json_reports(
        write_json=write_json,
        out_json=out_json,
        parquet_path=parquet_path,
        results=results,
        config=config,
        missing_error_prefix="Processed options chain not found",
    )
    if out_summary_json is not None:
        logger.info("QC summary written: %s", out_summary_json)
        logger.info("QC config written:  %s", out_config_json)

    passed, n_hard_fail, n_soft_fail, n_soft_warn = compute_outcome(results)
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
