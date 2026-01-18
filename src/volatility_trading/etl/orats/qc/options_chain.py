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
    """Filter to the region-of-interest (ROI) used by soft-check summaries."""
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )


# ------------------- Private QC Check Helpers -------------------
def _run_hard_checks(df: pl.DataFrame) -> list[QCCheckResult]:
    """Run hard (must-pass) checks on the full dataset."""
    results: list[QCCheckResult] = []
    hard_specs = [
        {
            "name": "keys_not_null",
            "df": df,
            "predicate_expr": expr_bad_null_keys(
                "trade_date", "expiry_date", "strike"
            ),
        },
        {
            "name": "trade_date_leq_expiry_date",
            "df": df,
            "predicate_expr": expr_bad_trade_after_expiry(),
        },
        {
            "name": "call_bid_ask_sane",
            "df": df.filter(pl.col("option_type") == "C"),
            "predicate_expr": expr_bad_bid_ask("bid_price", "ask_price"),
        },
        {
            "name": "put_bid_ask_sane",
            "df": df.filter(pl.col("option_type") == "P"),
            "predicate_expr": expr_bad_bid_ask("bid_price", "ask_price"),
        },
    ]
    for spec in hard_specs:
        results.append(
            run_hard_check(
                name=spec["name"],
                df=spec["df"],
                predicate_expr=spec["predicate_expr"],
                severity=Severity.HARD,
            )
        )
    return results


def _run_soft_checks(
        *,
        df: pl.DataFrame, 
        df_roi: pl.DataFrame, 
        config: QCConfig
) -> list[QCCheckResult]:
    """Run soft checks (arbitrage-style diagnostics) on GLOBAL and ROI subsets."""
    results: list[QCCheckResult] = []
    soft_thresholds = dict(config.soft_thresholds)
    soft_specs = [
        {
            "base_name": "strike_monotonicity",
            "flagger": flag_strike_monotonicity,
            "violation_col": "strike_monot_violation",
        },
        {
            "base_name": "maturity_monotonicity",
            "flagger": flag_maturity_monotonicity,
            "violation_col": "maturity_monot_violation",
        },
    ]

    for label, dfx in [("GLOBAL", df), ("ROI", df_roi)]:
        for spec in soft_specs:
            for opt in ["C", "P"]:
                results.append(
                    run_soft_check(
                        name=f"{label}_{spec['base_name']}_{opt}",
                        df=dfx,
                        flagger=spec["flagger"],
                        violation_col=spec["violation_col"],
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
    return results


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
    """Run QC on the processed options chain for one underlying.

    This is the main public entry point for validating a *processed* 
    options chain. It loads the WIDE chain from `proc_root`, converts it to 
    LONG format, runs a set of HARD (must-pass) checks globally, and SOFT 
    (diagnostic) checks on both:

    - GLOBAL: the entire chain
    - ROI: a restricted, more tradable region (DTE + |delta| band)

    HARD checks are intended to catch structural data issues 
    (e.g. missing keys, inverted bid/ask, trade_date after expiry). 
    SOFT checks are intended to measure the rate of common no-arbitrage 
    style issues (e.g. monotonicity violations).

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. ``"SPX"``). The loader will normalise it to
        uppercase and strip whitespace.
    proc_root:
        Root directory containing the processed options chain parquet(s).
        Expected layout is the one used by ``datasets.options_chain_path``.
    out_json:
        Optional explicit path for the QC summary JSON. If omitted and
        ``write_json=True``, the summary is written next to the parquet as
        ``qc_summary.json``.
    write_json:
        If True, write two sidecar files:

        - ``qc_summary.json``: list of check results
        - ``qc_config.json``: QC configuration used for the run

    dte_bins:
        Bin edges used by soft-check bucket summaries.
    delta_bins:
        Bin edges for ``|delta|`` used by soft-check bucket summaries.
    roi_dte_min, roi_dte_max:
        DTE bounds used to define the ROI subset.
    roi_delta_min, roi_delta_max:
        ``|delta|`` bounds used to define the ROI subset.
    top_k_buckets:
        Maximum number of worst buckets to include in soft-check details.

    Returns
    -------
    QCRunResult
        A structured report including:
        - the config used for this run
        - the list of check results
        - pass/fail summary counts
        - optional output JSON paths

    Notes
    -----
    - The QC module is *informational*: it does not drop rows or mutate the
      processed dataset.
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

    results: list[QCCheckResult] = []

    # ---- HARD checks (global) ----
    results.extend(_run_hard_checks(df))

    # ---- SOFT checks — report both Global and ROI ----
    df_roi = _apply_roi_filter(
        df,
        dte_min=config.roi_dte_min,
        dte_max=config.roi_dte_max,
        delta_min=config.roi_delta_min,
        delta_max=config.roi_delta_max,
    )
    n_rows_roi: int | None = int(df_roi.height)
    results.extend(_run_soft_checks(df=df, df_roi=df_roi, config=config))

    # ---- Report ----
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

    # Overall pass is strict: any failed check marks the run as failed.
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