"""
volatility_trading.etl.orats.qc.options_chain

QC runner for the processed ORATS *options chain* dataset.

This module is **read-only**: it does not drop or modify rows. It reports:
- **HARD checks**: must-pass invariants (null keys, trade_date <= expiry_date,
  bid/ask sanity: non-negative, not crossed).
- **SOFT checks**: diagnostics (strike/maturity monotonicity, locked/one-sided
  quotes, wide spreads).
- **INFO checks**: informal diagnostics (dondo nto fail).
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
    expr_bad_delta_bounds,
)
from .checks_info import (
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics,
)
from .checks_soft import (
    flag_iv_high,
    flag_locked_market,
    flag_maturity_monotonicity,
    flag_one_sided_quotes,
    flag_pos_vol_zero_oi,
    flag_strike_monotonicity,
    flag_theta_positive,
    flag_wide_spread,
    flag_zero_vol_pos_oi,
    flag_put_call_parity_mid_eu,
    flag_put_call_parity_tradable_eu,
    flag_put_call_parity_bounds_mid_am,
    flag_put_call_parity_bounds_tradable_am,
)
from .report import log_check, write_config_json, write_summary_json
from .runners import run_hard_check, run_info_check, run_soft_check
from .summarizers import summarize_by_bucket
from .types import Grade, QCCheckResult, QCConfig, QCRunResult, Severity
from volatility_trading.datasets import (
    options_chain_long_to_wide,
    options_chain_wide_to_long,
    options_chain_path,
    scan_options_chain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def _read_exercise_style_from_manifest(
    *,
    parquet_path: Path | None,
) -> str | None:
    """Read exercise_style from manifest.json next to the parquet file.

    Returns
    -------
    str | None
        "EU" or "AM" if valid, else None (missing/invalid/unreadable).
    """
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

    hard_specs = [
        # ---- Keys / dates ----
        {
            "name": "keys_not_null",
            "predicate_expr": expr_bad_null_keys("trade_date", "expiry_date", "strike"),
        },
        {
            "name": "trade_date_leq_expiry_date",
            "predicate_expr": expr_bad_trade_after_expiry(),
        },

        # ---- Quote diagnostics ----
        {
            "name": "bid_ask_sane",
            "predicate_expr": expr_bad_bid_ask("bid_price", "ask_price"),
        },
        {
            "name": "negative_quotes",
            "predicate_expr": expr_bad_negative_quotes("bid_price", "ask_price"),
        },
        {
            "name": "crossed_market",
            "predicate_expr": expr_bad_crossed_market("bid_price", "ask_price"),
        },

        # ---- Volume / OI diagnostics ----
        {
            "name": "negative_vol_oi",
            "predicate_expr": expr_bad_negative_vol_oi("volume", "open_interest"),
        },

        # ---- Greeks sign diagnostics ----
        {
            "name": "delta_bounds_sane",
            "predicate_expr": expr_bad_delta_bounds("delta", eps=1e-5),
        },
        {
            "name": "gamma_non_negative",
            "predicate_expr": expr_bad_negative("gamma", eps=1e-8),
        },
        {
            "name": "vega_non_negative",
            "predicate_expr": expr_bad_negative("vega", eps=1e-8),
        },

        # ---- IV sign diagnostics ----
        {
            "name": "iv_non_negative",
            "predicate_expr": expr_bad_negative("smoothed_iv", eps=1e-5),
        },
    ]

    for spec in hard_specs:
        results.append(
            run_hard_check(
                name=spec["name"],
                df=df,
                predicate_expr=spec["predicate_expr"],
                severity=Severity.HARD,
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
    """Run soft checks (diagnostic / arbitrage-style).

    Supports per-spec toggles:
    - use_roi: run GLOBAL + ROI (True) or GLOBAL only (False)
    - by_option_type: split into calls/puts or run once on the subset
    - requires_wide: build paired WIDE view (for PCP checks)
    """
    results: list[QCCheckResult] = []
    soft_thresholds = dict(config.soft_thresholds)

    # ---------------------------------------------------------------------
    # Base soft specs (always)
    # ---------------------------------------------------------------------
    soft_specs: list[dict] = [
        # ---- Arbitrage diagnostics ----
        {
            "base_name": "strike_monotonicity",
            "flagger": flag_strike_monotonicity,
            "violation_col": "strike_monot_violation",
            "flagger_kwargs": {"price_col": "mid_price"},
            "use_roi": True,
            "by_option_type": True,
        },
        {
            "base_name": "maturity_monotonicity",
            "flagger": flag_maturity_monotonicity,
            "violation_col": "maturity_monot_violation",
            "flagger_kwargs": {"price_col": "mid_price"},
            "use_roi": True,
            "by_option_type": True,
        },

        # ---- Quote diagnostics ----
        {
            "base_name": "locked_market",
            "flagger": flag_locked_market,
            "violation_col": "locked_market_violation",
            "flagger_kwargs": {},
            "use_roi": True,
            "by_option_type": True,
        },
        {
            "base_name": "one_sided_quotes",
            "flagger": flag_one_sided_quotes,
            "violation_col": "one_sided_quote_violation",
            "flagger_kwargs": {},
            "use_roi": True,
            "by_option_type": True,
        },
        {
            "base_name": "wide_spread",
            "flagger": flag_wide_spread,
            "violation_col": "wide_spread_violation",
            "flagger_kwargs": {"threshold": 1.0, "min_mid": 0.01},
            "use_roi": True,
            "by_option_type": True,
        },
        {
            "base_name": "very_wide_spread",
            "flagger": flag_wide_spread,
            "violation_col": "wide_spread_violation",
            "flagger_kwargs": {"threshold": 2.0, "min_mid": 0.01},
            "use_roi": True,
            "by_option_type": True,
        },

        # ---- Volume / OI diagnostics ----
        {
            "base_name": "zero_vol_pos_oi",
            "flagger": flag_zero_vol_pos_oi,
            "thresholds": {"mild": 0.05, "warn": 0.15, "fail": 0.30},
            "violation_col": "zero_vol_pos_oi_violation",
            "flagger_kwargs": {},
            "use_roi": True,
            "by_option_type": True,
        },
        {
            "base_name": "pos_vol_zero_oi",
            "flagger": flag_pos_vol_zero_oi,
            "thresholds": {"mild": 0.005, "warn": 0.02, "fail": 0.05},
            "violation_col": "pos_vol_zero_oi_violation",
            "flagger_kwargs": {},
            "use_roi": True,
            "by_option_type": True,
        },

        # ---- Greeks sign diagnostics ----
        {
            "base_name": "theta_positive",
            "flagger": flag_theta_positive,
            "thresholds": {"mild": 0.01, "warn": 0.03, "fail": 0.05},
            "violation_col": "theta_positive_violation",
            "flagger_kwargs": {"eps": 1e-8},
            "use_roi": True,
            "by_option_type": True,
        },

        # ---- IV diagnostics ----
        {
            "base_name": "high_iv",
            "flagger": flag_iv_high,
            "thresholds": {"mild": 0.01, "warn": 0.03, "fail": 0.05},
            "violation_col": "iv_too_high_violation",
            "flagger_kwargs": {"threshold": 1.0},
            "use_roi": False,
            "by_option_type": False,
        },
        {
            "base_name": "very_high_iv",
            "flagger": flag_iv_high,
            "thresholds": {"mild": 0.001, "warn": 0.005, "fail": 0.01},
            "violation_col": "iv_too_high_violation",
            "flagger_kwargs": {"threshold": 2.0},
            "use_roi": False,
            "by_option_type": False,
        },
    ]

    # ---------------------------------------------------------------------
    # Conditionally inject PCP specs
    # ---------------------------------------------------------------------
    if exercise_style == "EU":
        soft_specs.extend(
            [
                # ---- Put-call parity (EU) ----
                {
                    "base_name": "pcp_mid_eu",
                    "flagger": flag_put_call_parity_mid_eu,
                    "thresholds": {"mild": 0.01, "warn": 0.03, "fail": 0.05},
                    "violation_col": "pcp_mid_eu_violation",
                    "flagger_kwargs": {
                        "tol_floor": 0.01,
                    },
                    "use_roi": True,
                    "by_option_type": False,
                    "requires_wide": True,
                },
                {
                    "base_name": "pcp_tradable_eu",
                    "flagger": flag_put_call_parity_tradable_eu,
                    "thresholds": {"mild": 0.005, "warn": 0.02, "fail": 0.04},
                    "violation_col": "pcp_tradable_eu_violation",
                    "flagger_kwargs": {
                        "abs_tol": 0.01,
                    },
                    "use_roi": True,
                    "by_option_type": False,
                    "requires_wide": True,
                },
            ]
        )

    elif exercise_style == "AM":
        soft_specs.extend(
            [
                # ---- Put-call parity bounds (AM) ----
                {
                    "base_name": "pcp_bounds_mid_am",
                    "flagger": flag_put_call_parity_bounds_mid_am,
                    "thresholds": {"mild": 0.01, "warn": 0.03, "fail": 0.05},
                    "violation_col": "pcp_bounds_mid_am_violation",
                    "flagger_kwargs": {
                        "tol_floor": 0.01,
                    },
                    "use_roi": True,
                    "by_option_type": False,
                    "requires_wide": True,
                },
                {
                    "base_name": "pcp_bounds_tradable_am",
                    "flagger": flag_put_call_parity_bounds_tradable_am,
                    "thresholds": {"mild": 0.005, "warn": 0.02, "fail": 0.04},
                    "violation_col": "pcp_bounds_tradable_am_violation",
                    "flagger_kwargs": {
                        "abs_tol": 0.01,
                    },
                    "use_roi": True,
                    "by_option_type": False,
                    "requires_wide": True,
                },
            ]
        )

    # else: skip PCP checks entirely

    # ---------------------------------------------------------------------
    # Build WIDE views only if needed
    # ---------------------------------------------------------------------
    def _build_wide(df_long: pl.DataFrame) -> pl.DataFrame:
        """Build strict paired WIDE view from long (calls+puts on same row)."""
        wide = (
            options_chain_long_to_wide(
                long=df_long,
                how="inner",  # strict call+put pairing
            )
            .collect()
        )

        # summarize_by_bucket expects a "delta" column
        if "call_delta" in wide.columns:
            wide = wide.with_columns(pl.col("call_delta").alias("delta"))

        return wide

    needs_wide = any(
        bool(spec.get("requires_wide", False)) for spec in soft_specs
    )

    df_wide_global: pl.DataFrame | None = None
    df_wide_roi: pl.DataFrame | None = None

    if needs_wide:
        df_wide_global = _build_wide(df)
        df_wide_roi = _build_wide(df_roi)

    # ---------------------------------------------------------------------
    # Subset selection helper
    # ---------------------------------------------------------------------
    def _iter_subsets_for_spec(
        *,
        requires_wide: bool,
        use_roi: bool,
    ) -> list[tuple[str, pl.DataFrame]]:
        if requires_wide:
            if df_wide_global is None:
                return []
            out: list[tuple[str, pl.DataFrame]] = [("GLOBAL", df_wide_global)]
            if use_roi and df_wide_roi is not None:
                out.append(("ROI", df_wide_roi))
            return out

        out = [("GLOBAL", df)]
        if use_roi:
            out.append(("ROI", df_roi))
        return out

    # ---------------------------------------------------------------------
    # Run specs
    # ---------------------------------------------------------------------
    for spec in soft_specs:
        use_roi = bool(spec.get("use_roi", True))
        by_option_type = bool(spec.get("by_option_type", True))
        requires_wide = bool(spec.get("requires_wide", False))

        subsets = _iter_subsets_for_spec(
            requires_wide=requires_wide,
            use_roi=use_roi,
        )

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
                                **spec["flagger_kwargs"],
                            },
                            summarizer=summarize_by_bucket,
                            summarizer_kwargs={
                                "dte_bins": config.dte_bins,
                                "delta_bins": config.delta_bins,
                            },
                            thresholds=spec.get("thresholds", soft_thresholds),
                            severity=Severity.SOFT,
                            top_k_buckets=config.top_k_buckets,
                        )
                    )
            else:
                results.append(
                    run_soft_check(
                        name=f"{label}_{spec['base_name']}",
                        df=dfx,
                        flagger=spec["flagger"],
                        violation_col=spec["violation_col"],
                        flagger_kwargs=dict(spec["flagger_kwargs"]),
                        summarizer=summarize_by_bucket,
                        summarizer_kwargs={
                            "dte_bins": config.dte_bins,
                            "delta_bins": config.delta_bins,
                        },
                        thresholds=spec.get("thresholds", soft_thresholds),
                        severity=Severity.SOFT,
                        top_k_buckets=config.top_k_buckets,
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
    delta_bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
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
    exercise_style = _read_exercise_style_from_manifest(
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