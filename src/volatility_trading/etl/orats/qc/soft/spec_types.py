"""Spec dataclasses for SOFT QC row and dataset checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl


@dataclass(frozen=True, kw_only=True)
class SoftSpecBase:
    """Shared fields for all SOFT QC spec definitions."""

    base_name: str
    thresholds: dict[str, float] | None = None
    use_roi: bool = True
    sample_cols: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class SoftRowSpec(SoftSpecBase):
    """Definition of one row-level SOFT QC check."""

    flagger: Callable[..., pl.DataFrame]
    violation_col: str
    kind: Literal["row"] = "row"
    flagger_kwargs: dict[str, Any] = field(default_factory=dict)

    by_option_type: bool = True
    requires_wide: bool = False

    # If True, attach summarize_by_bucket(top buckets) to details.
    summarize_by_bucket: bool = True


@dataclass(frozen=True, kw_only=True)
class SoftDatasetSpec(SoftSpecBase):
    """Definition of one dataset-level SOFT QC check."""

    checker: Callable[..., dict[str, Any]]
    kind: Literal["dataset"] = "dataset"
    checker_kwargs: dict[str, Any] = field(default_factory=dict)


SoftSpec = SoftRowSpec | SoftDatasetSpec
