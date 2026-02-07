from __future__ import annotations

from ...soft.row_checks.greeks_iv import flag_iv_high
from ...soft.spec_types import SoftRowSpec
from ..specs_base import BASE_KEYS, IV_COLUMNS


def get_soft_specs() -> list[SoftRowSpec]:
    """Return SOFT (diagnostic) checks for daily-features QC."""
    specs: list[SoftRowSpec] = []

    for col in IV_COLUMNS:
        violation_col = f"{col}_too_high_violation"
        specs.append(
            SoftRowSpec(
                base_name=f"high_{col}",
                flagger=flag_iv_high,
                thresholds={"mild": 1e-3, "warn": 0.005, "fail": 0.01},
                violation_col=violation_col,
                flagger_kwargs={
                    "iv_col": col,
                    "threshold": 1.0,
                    "out_col": violation_col,
                },
                use_roi=False,
                by_option_type=False,
                sample_cols=BASE_KEYS + [col],
                summarize_by_bucket=False,
            )
        )

    return specs
