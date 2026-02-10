"""INFO QC spec definitions for options-chain panels."""

from __future__ import annotations

from ...info.spec_types import InfoSpec
from ...info.summarizers import (
    summarize_core_numeric_stats,
    summarize_risk_free_rate_metrics,
    summarize_volume_oi_metrics,
)

CORE_NUMERIC_COLS: tuple[str, ...] = (
    "underlying_price",
    "spot_price",
    "bid_price",
    "ask_price",
    "mid_price",
    "smoothed_iv",
    "strike",
    "dte",
    "delta",
    "gamma",
    "vega",
    "theta",
    "risk_free_rate",
    "dividend_yield",
    "volume",
    "open_interest",
)


def get_info_specs() -> list[InfoSpec]:
    """
    Define INFO checks. These are always-pass and only populate details.

    Keep these focused on descriptive metrics, not pass/fail logic.
    """
    return [
        InfoSpec(
            base_name="core_numeric_stats",
            summarizer=summarize_core_numeric_stats,
            summarizer_kwargs={"cols": list(CORE_NUMERIC_COLS)},
        ),
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
