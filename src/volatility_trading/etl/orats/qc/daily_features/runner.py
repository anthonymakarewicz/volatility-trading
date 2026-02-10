"""QC runner for processed ORATS daily-features panels.

Runs HARD, SOFT, and INFO suites and optionally writes JSON QC artifacts.
This module is read-only and does not mutate dataset rows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from volatility_trading.datasets import (
    daily_features_path,
    read_daily_features,
)

from ..common_helpers import (
    compute_outcome,
    run_all_checks,
    write_json_reports,
)
from ..reporting import log_check
from ..types import QCConfig, QCRunResult
from .hard.specs import get_hard_specs
from .info.specs import get_info_specs
from .soft.specs import get_soft_specs

logger = logging.getLogger(__name__)


def run_daily_features_qc(
    *,
    ticker: str,
    proc_root: Path | str,
    out_json: Path | str | None = None,
    write_json: bool = True,
) -> QCRunResult:
    """Run QC on one processed daily-features ticker panel.

    Args:
        ticker: Underlying ticker symbol.
        proc_root: Root directory containing processed daily-features panels.
        out_json: Optional explicit path for `qc_summary.json`.
        write_json: Write `qc_summary.json` and `qc_config.json` when `True`.

    Returns:
        Run summary with check results, outcome counts, timing, and artifact paths.
    """
    t0 = time.perf_counter()

    proc_root_p = Path(proc_root)
    ticker_s = str(ticker).strip().upper()

    parquet_path = daily_features_path(proc_root_p, ticker_s)

    config = QCConfig(
        ticker=ticker_s,
        run_roi=False,
    )

    df = read_daily_features(ticker=ticker_s, proc_root=proc_root_p)
    n_rows: int | None = int(df.height)

    logger.info(
        "Daily-features QC start ticker=%s rows=%s parquet=%s",
        ticker_s,
        n_rows,
        parquet_path,
    )

    results = run_all_checks(
        df_global=df,
        df_roi=df,
        config=config,
        exercise_style=None,
        hard_specs=get_hard_specs(),
        soft_specs=get_soft_specs(),
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
        missing_error_prefix="Processed daily features not found",
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
        n_rows_roi=None,
        n_hard_fail=n_hard_fail,
        n_soft_fail=n_soft_fail,
        n_soft_warn=n_soft_warn,
        out_summary_json=out_summary_json,
        out_config_json=out_config_json,
    )
