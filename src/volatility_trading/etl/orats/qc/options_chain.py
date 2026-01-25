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

from .checks_hard import (
    expr_bad_bid_ask,
    expr_bad_crossed_market,
    expr_bad_negative,
    expr_bad_negative_quotes,
    expr_bad_negative_vol_oi,
    expr_bad_null_keys,
    expr_bad_trade_after_expiry,
)
from .checks_info import (
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics,
)
from .report import log_check, write_config_json, write_summary_json
from .runners import run_hard_check, run_info_check, run_soft_check
from .specs_soft import get_soft_specs
from .summarizers import summarize_by_bucket
from .types import Grade, QCCheckResult, QCConfig, QCRunResult, Severity
from volatility_trading.datasets import (
    options_chain_long_to_wide,
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

def _run_hard_checks(df: pl.DataFrame) -> list[QCCheckResult]:
    """Run hard (must-pass) checks on the full dataset (GLOBAL)."""
    results: list[QCCheckResult] = []

    # Minimal columns always useful for debugging
    base_keys = [
        "trade_date", 
        "expiry_date", 
        "option_type", 
        "underlying_price", 
        "strike",
    ]

    hard_specs = [
        # ---- Keys / dates ----
        {
            "name": "keys_not_null",
            "predicate_expr": expr_bad_null_keys(
                "trade_date", "expiry_date", "strike"
            ),
            "sample_cols": base_keys,
        },
        {
            "name": "trade_date_leq_expiry_date",
            "predicate_expr": expr_bad_trade_after_expiry(),
            "sample_cols": ["trade_date", "expiry_date", "strike", "option_type"],
        },
        # ---- Quote diagnostics ----
        {
            "name": "bid_ask_sane",
            "predicate_expr": expr_bad_bid_ask("bid_price", "ask_price"),
            "sample_cols": base_keys + ["bid_price", "ask_price", "mid_price"],
        },
        {
            "name": "negative_quotes",
            "predicate_expr": expr_bad_negative_quotes("bid_price", "ask_price"),
            "sample_cols": base_keys + ["bid_price", "ask_price", "mid_price"],
        },
        {
            "name": "crossed_market",
            "predicate_expr": expr_bad_crossed_market("bid_price", "ask_price"),
            "sample_cols": base_keys + ["bid_price", "ask_price", "mid_price"],
        },
        # ---- Volume / OI diagnostics ----
        {
            "name": "negative_vol_oi",
            "predicate_expr": expr_bad_negative_vol_oi("volume", "open_interest"),
            "sample_cols": base_keys + ["volume", "open_interest"],
        },
        # ---- Greeks sign diagnostics ----
        {
            "name": "gamma_non_negative",
            "predicate_expr": expr_bad_negative("gamma", eps=1e-8),
            "sample_cols": base_keys + ["gamma"],
        },
        {
            "name": "vega_non_negative",
            "predicate_expr": expr_bad_negative("vega", eps=1e-8),
            "sample_cols": base_keys + ["vega"],
        },
        # ---- IV sign diagnostics ----
        {
            "name": "iv_non_negative",
            "predicate_expr": expr_bad_negative("smoothed_iv", eps=1e-5),
            "sample_cols": base_keys + ["smoothed_iv"],
        },
    ]

    for spec in hard_specs:
        results.append(
            run_hard_check(
                name=spec["name"],
                df=df,
                predicate_expr=spec["predicate_expr"],
                sample_n=10,
                sample_cols=spec.get("sample_cols"),
            )
        )

    return results


def _iter_subsets_for_spec(
    *,
    spec: dict,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    df_wide_global: pl.DataFrame | None,
    df_wide_roi: pl.DataFrame | None,
) -> list[tuple[str, pl.DataFrame]]:
    requires_wide = bool(spec.get("requires_wide", False))
    use_roi = bool(spec.get("use_roi", True))

    if requires_wide:
        if df_wide_global is None:
            return []
        out: list[tuple[str, pl.DataFrame]] = [("GLOBAL", df_wide_global)]
        if use_roi and df_wide_roi is not None:
            out.append(("ROI", df_wide_roi))
        return out

    out = [("GLOBAL", df_global)]
    if use_roi:
        out.append(("ROI", df_roi))
    return out


def _build_wide_views_if_needed(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    soft_specs: list[dict],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    def _build_wide(df_long: pl.DataFrame) -> pl.DataFrame:
        wide = (
            options_chain_long_to_wide(long=df_long, how="inner")
            .collect()
        )
        if "call_delta" in wide.columns:
            wide = wide.with_columns(pl.col("call_delta").alias("delta"))
        return wide

    needs_wide = any(
        bool(spec.get("requires_wide", False)) for spec in soft_specs
    )
    if not needs_wide:
        return None, None

    return _build_wide(df_global), _build_wide(df_roi)


def _run_soft_spec(
    *,
    spec: dict,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    df_wide_global: pl.DataFrame | None,
    df_wide_roi: pl.DataFrame | None,
    config: QCConfig,
    soft_thresholds: dict[str, float],
) -> list[QCCheckResult]:
    results: list[QCCheckResult] = []

    subsets = _iter_subsets_for_spec(
        spec=spec,
        df_global=df_global,
        df_roi=df_roi,
        df_wide_global=df_wide_global,
        df_wide_roi=df_wide_roi,
    )

    by_option_type = bool(spec.get("by_option_type", True))

    for label, dfx in subsets:
        if by_option_type:
            for opt in ["C", "P"]:
                results.append(
                    run_soft_check(
                        name=f"{label}_{spec['base_name']}_{opt}",
                        df=dfx,
                        flagger=spec["flagger"],
                        violation_col=spec["violation_col"],
                        flagger_kwargs={
                            "option_type": opt,
                            **spec.get("flagger_kwargs", {})
                        },
                        summarizer=summarize_by_bucket,
                        summarizer_kwargs={
                            "dte_bins": config.dte_bins,
                            "delta_bins": config.delta_bins,
                        },
                        thresholds=spec.get("thresholds", soft_thresholds),
                        top_k_buckets=config.top_k_buckets,
                        sample_cols=spec.get("sample_cols", None),
                        sample_n=5,
                    )
                )
        else:
            results.append(
                run_soft_check(
                    name=f"{label}_{spec['base_name']}",
                    df=dfx,
                    flagger=spec["flagger"],
                    violation_col=spec["violation_col"],
                    flagger_kwargs=dict(spec.get("flagger_kwargs", {})),
                    summarizer=summarize_by_bucket,
                    summarizer_kwargs={
                        "dte_bins": config.dte_bins,
                        "delta_bins": config.delta_bins,
                    },
                    thresholds=spec.get("thresholds", soft_thresholds),
                    top_k_buckets=config.top_k_buckets,
                    sample_cols=spec.get("sample_cols", None),
                    sample_n=5,
                )
            )

    return results


def _run_soft_checks(
    *,
    df: pl.DataFrame,
    df_roi: pl.DataFrame,
    config: QCConfig,
    exercise_style: str | None,
) -> list[QCCheckResult]:
    results: list[QCCheckResult] = []
    soft_thresholds = dict(config.soft_thresholds)

    soft_specs = get_soft_specs(
        exercise_style=exercise_style,
        config=config,
    )

    df_wide_global, df_wide_roi = _build_wide_views_if_needed(
        df_global=df,
        df_roi=df_roi,
        soft_specs=soft_specs,
    )

    for spec in soft_specs:
        results.extend(
            _run_soft_spec(
                spec=spec,
                df_global=df,
                df_roi=df_roi,
                df_wide_global=df_wide_global,
                df_wide_roi=df_wide_roi,
                config=config,
                soft_thresholds=soft_thresholds,
            )
        )

    return results


def _run_info_checks(
    *,
    df: pl.DataFrame,
    df_roi: pl.DataFrame,
) -> list[QCCheckResult]:
    """Run informational checks (always pass) and store metrics in details."""
    results: list[QCCheckResult] = []

    for label, dfx in [("GLOBAL", df), ("ROI", df_roi)]:
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

    results: list[QCCheckResult] = []
    results.extend(_run_hard_checks(df))
    results.extend(
        _run_soft_checks(
            df=df,
            df_roi=df_roi,
            config=config,
            exercise_style=exercise_style,
        )
    )
    results.extend(_run_info_checks(df=df, df_roi=df_roi))

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