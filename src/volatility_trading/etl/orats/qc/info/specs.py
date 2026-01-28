# qc/info/specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .summarizers import (
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics
)


@dataclass(frozen=True)
class InfoSpec:
    base_name: str
    summarizer: Callable[..., dict[str, Any]]
    summarizer_kwargs: dict[str, Any] | None = None


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