from __future__ import annotations

from volatility_trading.etl.orats.processed.daily_features.config import (
    DAILY_FEATURES_CORE_COLUMNS,
)

from ..hard.exprs import expr_bad_negative, expr_bad_null_keys
from ..hard.spec_types import HardSpec
from ..soft.row_checks.greeks_iv import flag_iv_high
from ..soft.spec_types import SoftRowSpec


BASE_KEYS = [
    "ticker",
    "trade_date",
]

IV_COLUMNS = tuple(
    col for col in DAILY_FEATURES_CORE_COLUMNS if col.startswith("iv_")
)
HV_COLUMNS = tuple(
    col for col in DAILY_FEATURES_CORE_COLUMNS if col.startswith("hv_")
)


def get_hard_specs() -> list[HardSpec]:
    """Return HARD (must-pass) checks for daily-features QC."""
    specs: list[HardSpec] = [
        HardSpec(
            name="keys_not_null",
            predicate_expr=expr_bad_null_keys(*BASE_KEYS),
            sample_cols=BASE_KEYS,
        ),
    ]

    for col in (*IV_COLUMNS, *HV_COLUMNS):
        specs.append(
            HardSpec(
                name=f"{col}_non_negative",
                predicate_expr=expr_bad_negative(col, eps=1e-12),
                sample_cols=BASE_KEYS + [col],
            )
        )

    return specs


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
