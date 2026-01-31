# qc/soft/suite.py
from __future__ import annotations

import polars as pl

from ..runners import run_soft_check, run_soft_check_dataset
from .specs import get_soft_specs
from .summarizers import summarize_by_bucket
from ..types import QCCheckResult, QCConfig
from .utils import build_wide_views_if_needed, iter_subsets_for_spec


def run_soft_suite(
    *,
    df_global: pl.DataFrame,
    df_roi: pl.DataFrame,
    config: QCConfig,
    exercise_style: str | None,
) -> list[QCCheckResult]:
    """
    Run SOFT checks for the options chain QC.

    Supports:
      - Row-level specs (flagger -> boolean violation_col)
      - Dataset-level specs (checker -> metrics dict)

    Orchestration:
      - builds WIDE only if needed
      - routes GLOBAL/ROI subsets per spec
      - optionally splits row checks by option_type (C/P)
    """
    results: list[QCCheckResult] = []
    soft_thresholds = dict(config.soft_thresholds)

    soft_specs = get_soft_specs(exercise_style=exercise_style)

    df_wide_global, df_wide_roi = build_wide_views_if_needed(
        df_global=df_global,
        df_roi=df_roi,
        soft_specs=soft_specs,
    )

    for spec in soft_specs:
        subsets = iter_subsets_for_spec(
            spec=spec,
            df_global=df_global,
            df_roi=df_roi,
            df_wide_global=df_wide_global,
            df_wide_roi=df_wide_roi,
        )

        for label, dfx in subsets:
            # -----------------------------------------------------------------
            # Dataset-level checks
            # -----------------------------------------------------------------
            if spec.kind == "dataset":
                results.append(
                    run_soft_check_dataset(
                        name=f"{label}_{spec.base_name}",
                        df=dfx,
                        checker=spec.checker,
                        checker_kwargs=dict(spec.checker_kwargs),
                        thresholds=spec.thresholds or soft_thresholds,
                    )
                )
                continue

            # -----------------------------------------------------------------
            # Row-level checks
            # -----------------------------------------------------------------
            if spec.by_option_type:
                for opt in ["C", "P"]:
                    results.append(
                        run_soft_check(
                            name=f"{label}_{spec.base_name}_{opt}",
                            df=dfx,
                            flagger=spec.flagger,
                            violation_col=spec.violation_col,
                            flagger_kwargs={
                                "option_type": opt,
                                **dict(spec.flagger_kwargs),
                            },
                            summarizer=summarize_by_bucket,
                            summarizer_kwargs={
                                "dte_bins": config.dte_bins,
                                "delta_bins": config.delta_bins,
                            },
                            thresholds=spec.thresholds or soft_thresholds,
                            top_k_buckets=config.top_k_buckets,
                            sample_cols=spec.sample_cols,
                            sample_n=5,
                        )
                    )
            else:
                results.append(
                    run_soft_check(
                        name=f"{label}_{spec.base_name}",
                        df=dfx,
                        flagger=spec.flagger,
                        violation_col=spec.violation_col,
                        flagger_kwargs=dict(spec.flagger_kwargs),
                        summarizer=summarize_by_bucket,
                        summarizer_kwargs={
                            "dte_bins": config.dte_bins,
                            "delta_bins": config.delta_bins,
                        },
                        thresholds=spec.thresholds or soft_thresholds,
                        top_k_buckets=config.top_k_buckets,
                        sample_cols=spec.sample_cols,
                        sample_n=5,
                    )
                )

    return results