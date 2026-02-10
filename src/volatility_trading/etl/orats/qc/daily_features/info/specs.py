"""INFO QC spec definitions for daily-features panels."""

from __future__ import annotations

from ...info.spec_types import InfoSpec
from ...info.summarizers import summarize_core_numeric_stats
from ..specs_base import INFO_COLUMNS


def get_info_specs() -> list[InfoSpec]:
    """Define INFO checks for daily-features QC."""
    return [
        InfoSpec(
            base_name="core_numeric_stats",
            summarizer=summarize_core_numeric_stats,
            summarizer_kwargs={"cols": list(INFO_COLUMNS)},
        ),
    ]
