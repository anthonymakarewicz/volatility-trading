# qc/info/suite.py
from __future__ import annotations

import polars as pl

from ..runners import run_info_check
from ..types import QCCheckResult
from .specs import get_info_specs


def run_info_suite(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
) -> list[QCCheckResult]:
    """
    Run INFO checks (always pass).

    We run the same INFO specs on:
      - GLOBAL dataset
      - ROI subset
    """
    results: list[QCCheckResult] = []
    specs = get_info_specs()

    for label, dfx in [("GLOBAL", df_global), ("ROI", df_roi)]:
        for spec in specs:
            results.append(
                run_info_check(
                    name=f"{label}_{spec.base_name}",
                    df=dfx,
                    summarizer=spec.summarizer,
                    summarizer_kwargs=dict(spec.summarizer_kwargs or {}),
                )
            )

    return results