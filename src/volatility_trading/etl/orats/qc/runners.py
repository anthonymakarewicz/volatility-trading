from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from .types import Grade, QCCheckResult, Severity


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _grade_from_thresholds(rate: float, thresholds: dict[str, float]) -> Grade:
    """
    thresholds example:
        {"mild": 0.01, "warn": 0.03, "fail": 0.10}
    """
    mild = thresholds.get("mild", 0.0)
    warn = thresholds.get("warn", 1.0)
    fail = thresholds.get("fail", 1.0)

    if rate >= fail:
        return Grade.FAIL
    if rate >= warn:
        return Grade.WARN
    if rate >= mild:
        return Grade.MILD
    return Grade.OK


def _count_bool_true(s: pl.Series) -> int:
    # Polars booleans can include null -> treat null as False
    return int(s.fill_null(False).sum())


# -----------------------------------------------------------------------------
# Public runners
# -----------------------------------------------------------------------------

def run_hard_check(
    *,
    name: str,
    df: pl.DataFrame,
    predicate_expr: pl.Expr,
    severity: Severity = Severity.HARD,
    allow_rate: float = 0.0,
    details: dict[str, Any] | None = None,
) -> QCCheckResult:
    """
    Hard check runner: predicate_expr flags "bad rows".
    We compute n_bad and viol_rate and grade as OK/FAIL.

    If allow_rate > 0, you allow tiny percentages to pass (rare).
    """
    n_rows = int(df.height)
    if n_rows == 0:
        return QCCheckResult(
            name=name,
            severity=severity,
            grade=Grade.FAIL,
            passed=False,
            n_rows=0,
            n_viol=0,
            viol_rate=None,
            details={"reason": "empty dataframe", **(details or {})},
        )

    bad_mask = df.select(predicate_expr.fill_null(False).alias("_bad"))["_bad"]
    n_bad = _count_bool_true(bad_mask)
    rate = n_bad / n_rows

    passed = rate <= allow_rate
    grade = Grade.OK if passed else Grade.FAIL

    out_details = dict(details or {})
    return QCCheckResult(
        name=name,
        severity=severity,
        grade=grade,
        passed=passed,
        n_rows=n_rows,
        n_viol=n_bad,
        viol_rate=rate,
        details=out_details,
    )


def run_soft_check(
    *,
    name: str,
    df: pl.DataFrame,
    flagger: Callable[..., pl.DataFrame],
    violation_col: str,
    summarizer: Callable[..., pl.DataFrame] | None = None,
    thresholds: dict[str, float] | None = None,
    severity: Severity = Severity.SOFT,
    details: dict[str, Any] | None = None,
    summarizer_kwargs: dict[str, Any] | None = None,
    flagger_kwargs: dict[str, Any] | None = None,
    top_k_buckets: int = 10,
) -> QCCheckResult:
    """
    Soft check runner:
      - flagger(df, **kwargs) returns df with boolean `violation_col`
      - summarizer(df_flagged, **kwargs) optionally returns bucket table
    """
    thresholds = thresholds or {"mild": 0.01, "warn": 0.03, "fail": 0.10}
    summarizer_kwargs = summarizer_kwargs or {}
    flagger_kwargs = flagger_kwargs or {}

    n_rows = int(df.height)
    if n_rows == 0:
        return QCCheckResult(
            name=name,
            severity=severity,
            grade=Grade.FAIL,
            passed=False,
            n_rows=0,
            n_viol=0,
            viol_rate=None,
            details={"reason": "empty dataframe", **(details or {})},
        )

    flagged = flagger(df, **flagger_kwargs)

    if violation_col not in flagged.columns:
        raise ValueError(
            f"flagger did not produce expected violation_col={violation_col!r}"
        )

    n_viol = _count_bool_true(flagged[violation_col])
    rate = n_viol / n_rows

    grade = _grade_from_thresholds(rate, thresholds)
    passed = grade in {Grade.OK, Grade.MILD}  # typical policy

    out_details = dict(details or {})
    out_details["thresholds"] = thresholds

    # Optional bucket summary (top-K)
    if summarizer is not None:
        summary = summarizer(
            flagged, 
            violation_col=violation_col, 
            **summarizer_kwargs
        )
        if summary.height > 0:
            out_details["top_buckets"] = summary.head(top_k_buckets).to_dicts()

    return QCCheckResult(
        name=name,
        severity=severity,
        grade=grade,
        passed=passed,
        n_rows=n_rows,
        n_viol=n_viol,
        viol_rate=rate,
        details=out_details,
    )


def run_info_check(
    *,
    name: str,
    df: pl.DataFrame,
    summarizer: Callable[..., dict[str, Any]],
    summarizer_kwargs: dict[str, Any] | None = None,
    severity: Severity = Severity.INFO,
) -> QCCheckResult:
    """
    Run an informational check (always-pass). Stores metrics in details.
    """
    summarizer_kwargs = summarizer_kwargs or {}

    n_rows = int(df.height)
    details = summarizer(df=df, **summarizer_kwargs)

    return QCCheckResult(
        name=name,
        severity=severity,
        grade=Grade.OK,
        passed=True,
        n_rows=n_rows,
        n_viol=None,
        viol_rate=None,
        details=details,
    )