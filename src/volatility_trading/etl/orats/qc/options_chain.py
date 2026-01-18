from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .report import write_summary_json, write_config_json, log_check
from .runners import run_hard_check, run_soft_check
from .checks_hard import (
    expr_bad_null_keys,
    expr_bad_bid_ask,
    expr_bad_trade_after_expiry,
)
from .checks_soft import (
    flag_strike_monotonicity,
    flag_maturity_monotonicity,
    summarize_by_bucket,
)
from .types import (
    Severity, 
    Grade, 
    QCConfig, 
    QCRunResult, 
    QCCheckResult
)
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    scan_options_chain,
    options_chain_path,
)

logger = logging.getLogger(__name__)


def _apply_roi_filter(
    df: pl.DataFrame,
    *,
    dte_min: int = 10,
    dte_max: int = 60,
    delta_min: float = 0.1,
    delta_max: float = 0.9,
) -> pl.DataFrame:
    """Return the region-of-interest subset used for soft-check reporting."""
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )


def run_qc(
    *,
    ticker: str,
    proc_root: Path | str,
    out_json: Path | str | None = None,
    write_json: bool = True,
    dte_bins: Sequence[int | float] = (0, 10, 30, 60, 180),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    roi_dte_min: int = 10,
    roi_dte_max: int = 60,
    roi_delta_min: float = 0.1,
    roi_delta_max: float = 0.9,
    top_k_buckets: int = 10,
) -> QCRunResult:
    """Run QC checks on the processed options chain for a single ticker.

    Notes
    -----
    - Loads processed WIDE options chain via `datasets.scan_options_chain()`.
    - Converts to LONG with `datasets.options_chain_wide_to_long()`.
    - Runs HARD checks globally and SOFT checks on both GLOBAL + ROI regions.
    - Optionally writes two sidecars next to the parquet:
        * qc_summary.json (results)
        * qc_config.json  (run configuration)

    Returns
    -------
    QCRunResult
        Container with the list of check results and output paths.
    """
    t0 = time.perf_counter()

    proc_root_p = Path(proc_root)
    ticker_s = str(ticker).strip().upper()

    parquet_path: Path | None = None
    try:
        parquet_path = options_chain_path(proc_root_p, ticker_s)
    except Exception:
        parquet_path = None

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

    logger.info("QC loading ticker=%s", ticker_s)

    lf = scan_options_chain(ticker_s, proc_root=proc_root_p)
    df = options_chain_wide_to_long(lf).collect()

    n_rows: int | None = int(df.height)

    results: list[QCCheckResult] = []

    # ---------------------------------------------------------------------
    # HARD checks (global)
    # ---------------------------------------------------------------------
    results.append(
        run_hard_check(
            name="keys_not_null",
            df=df,
            predicate_expr=expr_bad_null_keys(
                "trade_date", "expiry_date", "strike"
            ),
            severity=Severity.HARD,
        )
    )
    results.append(
        run_hard_check(
            name="trade_date_leq_expiry_date",
            df=df,
            predicate_expr=expr_bad_trade_after_expiry(),
            severity=Severity.HARD,
        )
    )
    results.append(
        run_hard_check(
            name="call_bid_ask_sane",
            df=df.filter(pl.col("option_type") == "C"),
            predicate_expr=expr_bad_bid_ask("bid_price", "ask_price"),
            severity=Severity.HARD,
        )
    )
    results.append(
        run_hard_check(
            name="put_bid_ask_sane",
            df=df.filter(pl.col("option_type") == "P"),
            predicate_expr=expr_bad_bid_ask("bid_price", "ask_price"),
            severity=Severity.HARD,
        )
    )

    # ---------------------------------------------------------------------
    # SOFT checks — report both Global and ROI
    # ---------------------------------------------------------------------
    df_roi = _apply_roi_filter(
        df,
        dte_min=config.roi_dte_min,
        dte_max=config.roi_dte_max,
        delta_min=config.roi_delta_min,
        delta_max=config.roi_delta_max,
    )

    n_rows_roi: int | None = int(df_roi.height)

    soft_thresholds = dict(config.soft_thresholds)

    for label, dfx in [("GLOBAL", df), ("ROI", df_roi)]:
        # Strike monotonicity
        for opt in ["C", "P"]:
            results.append(
                run_soft_check(
                    name=f"{label}_strike_monotonicity_{opt}",
                    df=dfx,
                    flagger=flag_strike_monotonicity,
                    violation_col="strike_monot_violation",
                    flagger_kwargs={
                        "option_type": opt,
                        "price_col": "mid_price",
                    },
                    summarizer=summarize_by_bucket,
                    summarizer_kwargs={
                        "dte_bins": config.dte_bins,
                        "delta_bins": config.delta_bins,
                    },
                    thresholds=soft_thresholds,
                    severity=Severity.SOFT,
                    top_k_buckets=config.top_k_buckets,
                )
            )

        # Maturity monotonicity
        for opt in ["C", "P"]:
            results.append(
                run_soft_check(
                    name=f"{label}_maturity_monotonicity_{opt}",
                    df=dfx,
                    flagger=flag_maturity_monotonicity,
                    violation_col="maturity_monot_violation",
                    flagger_kwargs={
                        "option_type": opt,
                        "price_col": "mid_price",
                    },
                    summarizer=summarize_by_bucket,
                    summarizer_kwargs={
                        "dte_bins": config.dte_bins,
                        "delta_bins": config.delta_bins,
                    },
                    thresholds=soft_thresholds,
                    severity=Severity.SOFT,
                    top_k_buckets=config.top_k_buckets,
                )
            )

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    for r in results:
        log_check(logger, r)

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

    passed = all(r.passed for r in results)

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