# soft/suite.py
from __future__ import annotations

import polars as pl

from ..runners import run_soft_check
from ..specs_soft import get_soft_specs
from ..summarizers import summarize_by_bucket
from ..types import QCCheckResult, QCConfig

from .utils import _iter_subsets_for_spec, _build_wide_views_if_needed


def run_soft_suite(
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
        subsets = _iter_subsets_for_spec(
            spec=spec,
            df_global=df,
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
                                **spec.get("flagger_kwargs", {}),
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