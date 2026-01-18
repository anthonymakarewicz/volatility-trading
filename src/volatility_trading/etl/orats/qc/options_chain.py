from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .types import Severity
from .runners import run_hard_check, run_soft_check
from .report import write_summary_json, log_check
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
    return df.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("delta").abs().is_between(delta_min, delta_max),
    )


def run_qc(
    *,
    proc_root: Path | str,
    ticker: str,
    out_json: Path | str | None = None,
    write_json: bool = True,
    dte_bins: Sequence[int | float] = (0, 10, 30, 60, 180),
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    roi_dte_min: int = 10,
    roi_dte_max: int = 60,
    roi_delta_min: float = 0.1,
    roi_delta_max: float = 0.9,
    top_k_buckets: int = 10,
) -> list:
    """
    Run QC checks on a processed options chain for one ticker.

    This function takes only the processed *root* and constructs the data path:
        proc_root / underlying={ticker} / part-0000.parquet
    (or detects a partitioned parquet layout under the same directory).

    Returns
    -------
    list[QCCheckResult]
        List of QC results (hard + soft checks).
    """
    logger.info("QC loading ticker=%s", ticker)

    proc_root = Path(proc_root)
    lf = scan_options_chain(ticker, proc_root=proc_root)
    df = options_chain_wide_to_long(lf).collect()

    results = []
    # ---- HARD checks (global) ----
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

    # ---- SOFT checks — report both Global and ROI ----
    df_roi = _apply_roi_filter(
        df,
        dte_min=roi_dte_min,
        dte_max=roi_dte_max,
        delta_min=roi_delta_min,
        delta_max=roi_delta_max,
    )

    soft_thresholds = {"mild": 0.01, "warn": 0.03, "fail": 0.10}

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
                        "option_type": opt, "price_col": "mid_price"
                    },
                    summarizer=summarize_by_bucket,
                    summarizer_kwargs={
                        "dte_bins": dte_bins, "delta_bins": delta_bins
                    },
                    thresholds=soft_thresholds,
                    severity=Severity.SOFT,
                    top_k_buckets=top_k_buckets,
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
                        "option_type": opt, "price_col": "mid_price"
                    },
                    summarizer=summarize_by_bucket,
                    summarizer_kwargs={
                        "dte_bins": dte_bins, "delta_bins": delta_bins
                    },
                    thresholds=soft_thresholds,
                    severity=Severity.SOFT,
                    top_k_buckets=top_k_buckets,
                )
            )

    # ---- Report + persist ----
    for r in results:
        log_check(logger, r)

    if write_json:
        if out_json is None:
            path = options_chain_path(proc_root, ticker)
            if not path.exists():
                raise FileNotFoundError(
                    f"Processed options chain not found: {path}"
                )
            out_json_p = path.parent / "qc_summary.json"
        else:
            out_json_p = Path(out_json)

        write_summary_json(out_json_p, results)
        logger.info("QC summary written: %s", out_json_p)

    return results