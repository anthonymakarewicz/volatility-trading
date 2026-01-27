from __future__ import annotations

from collections.abc import Callable, Sequence
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


def _make_jsonable_sample(df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a small Polars df into JSON-safe list[dict].

    Polars may return python date/datetime objects which vanilla json can't
    serialize. We cast temporal columns to Utf8 for safety.
    """
    if df.height == 0:
        return []

    out = df
    for col, dtype in out.schema.items():
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            out = out.with_columns(pl.col(col).cast(pl.Utf8))

    return out.to_dicts()


# -----------------------------------------------------------------------------
# Public runners
# -----------------------------------------------------------------------------

def run_hard_check(
    *,
    name: str,
    df: pl.DataFrame,
    predicate_expr: pl.Expr,
    allow_rate: float = 0.0,
    details: dict[str, Any] | None = None,
    sample_n: int = 0,
    sample_cols: Sequence[str] | None = None,
) -> QCCheckResult:
    """
    Hard check runner: predicate_expr flags "bad rows".
    We compute n_bad and viol_rate and grade as OK/FAIL.

    If allow_rate > 0, you allow tiny percentages to pass (rare).

    Optional:
    - sample_n: if > 0 and violations exist, attach sample rows to details
    - sample_cols: restrict sample schema (recommended)
    """
    n_rows = int(df.height)
    if n_rows == 0:
        return QCCheckResult(
            name=name,
            severity=Severity.HARD,
            grade=Grade.FAIL,
            passed=False,
            n_rows=0,
            n_viol=0,
            viol_rate=None,
            details={"reason": "empty dataframe", **(details or {})},
        )

    bad_expr = predicate_expr.fill_null(False)
    bad_mask = df.select(bad_expr.alias("_bad"))["_bad"]
    n_bad = _count_bool_true(bad_mask)
    rate = n_bad / n_rows

    passed = rate <= allow_rate
    grade = Grade.OK if passed else Grade.FAIL

    out_details = dict(details or {})

    # Attach a small sample of violating rows only if failures exist
    if sample_n > 0 and n_bad > 0:
        # Filter the violating rows
        bad_df = df.filter(bad_expr).head(sample_n)

        if sample_cols is not None:
            cols = [c for c in sample_cols if c in bad_df.columns]
            if cols:
                bad_df = bad_df.select(cols)

        out_details["sample_n"] = int(min(sample_n, n_bad))
        out_details["sample_rows"] = _make_jsonable_sample(bad_df)

    return QCCheckResult(
        name=name,
        severity=Severity.HARD,
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
    details: dict[str, Any] | None = None,
    summarizer_kwargs: dict[str, Any] | None = None,
    flagger_kwargs: dict[str, Any] | None = None,
    top_k_buckets: int = 10,
    sample_n: int = 0,
    sample_cols: Sequence[str] | None = None,
    sample_when_grade_at_least: Grade = Grade.WARN,
) -> QCCheckResult:
    """
    Soft check runner:
      - flagger(df, **kwargs) returns df with boolean `violation_col`
      - summarizer(df_flagged, **kwargs) optionally returns bucket table

    Optional:
    - sample_n: attach sample violating rows into details
    - sample_cols: restrict sample schema
    - sample_when_grade_at_least: only attach sample when grade >= WARN (recommended)
    """
    thresholds = thresholds or {"mild": 0.01, "warn": 0.03, "fail": 0.10}
    summarizer_kwargs = summarizer_kwargs or {}
    flagger_kwargs = flagger_kwargs or {}

    n_rows = int(df.height)
    if n_rows == 0:
        return QCCheckResult(
            name=name,
            severity=Severity.SOFT,
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
    passed = grade in {Grade.OK, Grade.MILD}

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

    # Optional: attach violating row samples (only if needed)
    if sample_n > 0 and n_viol > 0:
        # only sample when grade is "bad enough"
        grade_rank = {Grade.OK: 0, Grade.MILD: 1, Grade.WARN: 2, Grade.FAIL: 3}
        if grade_rank[grade] >= grade_rank[sample_when_grade_at_least]:
            viol_df = flagged.filter(
                pl.col(violation_col).fill_null(False)
            ).head(sample_n)

            if sample_cols is not None:
                cols = [c for c in sample_cols if c in viol_df.columns]
                if cols:
                    viol_df = viol_df.select(cols)

            out_details["sample_n"] = int(min(sample_n, n_viol))
            out_details["sample_rows"] = _make_jsonable_sample(viol_df)

    return QCCheckResult(
        name=name,
        severity=Severity.SOFT,
        grade=grade,
        passed=passed,
        n_rows=n_rows,
        n_viol=n_viol,
        viol_rate=rate,
        details=out_details,
    )


def run_soft_check_dataset(
    *,
    name: str,
    df: pl.DataFrame,
    checker: Callable[..., dict[str, Any]],
    thresholds: dict[str, float] | None = None,
    checker_kwargs: dict[str, Any] | None = None,
    metric_key: str = "viol_rate",
    details: dict[str, Any] | None = None,
) -> QCCheckResult:
    """
    Dataset-level soft check runner.

    `checker(df, **kwargs) -> dict` must return at least:
      - metric_key (default "viol_rate") : float
    Optional keys:
      - n_viol: int
      - n_units: int  (e.g., number of sessions/days examined)
    """
    thresholds = thresholds or {"mild": 0.01, "warn": 0.03, "fail": 0.10}
    checker_kwargs = checker_kwargs or {}

    n_rows = int(df.height)
    if n_rows == 0:
        return QCCheckResult(
            name=name,
            severity=Severity.SOFT,
            grade=Grade.FAIL,
            passed=False,
            n_rows=0,
            n_units=0,
            n_viol=0,
            viol_rate=None,
            details={"reason": "empty dataframe", **(details or {})},
        )

    out = checker(df=df, **checker_kwargs)

    if metric_key not in out:
        raise ValueError(
            f"dataset checker must return metric_key={metric_key!r}; "
            f"got keys={list(out.keys())}"
        )

    metric = out[metric_key]
    if metric is None:
        raise ValueError(
            f"dataset checker returned None metric for {metric_key!r}"
        )

    grade = _grade_from_thresholds(float(metric), thresholds)
    passed = grade in {Grade.OK, Grade.MILD}

    # Standardized metrics (prefer checker-provided values)
    n_viol = int(out["n_viol"]) if out.get("n_viol") is not None else None
    n_units = int(out["n_units"]) if out.get("n_units") is not None else None

    # Keep only checker-specific context (plus thresholds).
    drop_keys = {metric_key, "viol_rate", "n_units", "n_viol"}
    checker_details = {k: v for k, v in out.items() if k not in drop_keys}

    out_details = dict(details or {})
    out_details["thresholds"] = thresholds
    out_details.update(checker_details)

    return QCCheckResult(
        name=name,
        severity=Severity.SOFT,
        grade=grade,
        passed=passed,
        n_rows=n_rows,       # actual df rows used
        n_units=n_units,     # units examined (sessions/days/groups)
        n_viol=n_viol,       # units violated
        viol_rate=float(metric),
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