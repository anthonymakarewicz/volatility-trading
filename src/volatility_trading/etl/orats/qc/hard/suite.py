"""Execution suite for HARD QC checks."""

from __future__ import annotations

import polars as pl

from ..runners import run_hard_check
from ..types import QCCheckResult
from .spec_types import HardSpec


def run_hard_suite(
    *,
    df_global: pl.DataFrame,
    hard_specs: list[HardSpec],
) -> list[QCCheckResult]:
    """Run HARD (must-pass) checks on the full dataset (GLOBAL)."""
    results: list[QCCheckResult] = []

    for spec in hard_specs:
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
