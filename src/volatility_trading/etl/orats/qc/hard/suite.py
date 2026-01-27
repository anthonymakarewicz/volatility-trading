from __future__ import annotations

import polars as pl

from ..runners import run_hard_check
from ..types import QCCheckResult
from .specs import get_hard_specs


def run_hard_suite(*, df_global: pl.DataFrame) -> list[QCCheckResult]:
    """Run HARD (must-pass) checks on the full dataset (GLOBAL)."""
    results: list[QCCheckResult] = []

    for spec in get_hard_specs():
        results.append(
            run_hard_check(
                name=spec.name,
                df=df_global,
                predicate_expr=spec.predicate_expr,
                sample_n=spec.sample_n,
                sample_cols=spec.sample_cols,
            )
        )

    return results