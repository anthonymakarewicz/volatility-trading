# qc/info/specs.py
from __future__ import annotations

from .spec_types import InfoSpec

from .summarizers import (
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics
)


def get_info_specs() -> list[InfoSpec]:
    """
    Define INFO checks. These are always-pass and only populate details.

    Keep these focused on descriptive metrics, not pass/fail logic.
    """
    return [
        InfoSpec(
            base_name="volume_oi_metrics",
            summarizer=summarize_volume_oi_metrics,
            summarizer_kwargs=None,
        ),
        InfoSpec(
            base_name="risk_free_rate_metrics",
            summarizer=summarize_risk_free_rate_metrics,
            summarizer_kwargs=None,
        ),
    ]