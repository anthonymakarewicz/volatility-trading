from __future__ import annotations

from ...hard.exprs import expr_bad_negative, expr_bad_null_keys
from ...hard.spec_types import HardSpec
from ..specs_base import BASE_KEYS, HV_COLUMNS, IV_COLUMNS


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
