# qc/spec_types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import polars as pl


@dataclass(frozen=True)
class SoftSpecBase:
    base_name: str
    thresholds: dict[str, float] | None = None
    use_roi: bool = True
    sample_cols: list[str] | None = None


@dataclass(frozen=True)
class SoftRowSpec(SoftSpecBase):
    kind: Literal["row"] = "row"
    flagger: Callable[..., pl.DataFrame] = None  # required in practice
    violation_col: str = ""                      # required in practice
    flagger_kwargs: dict[str, Any] = field(default_factory=dict)
    by_option_type: bool = True
    requires_wide: bool = False


@dataclass(frozen=True)
class SoftDatasetSpec(SoftSpecBase):
    kind: Literal["dataset"] = "dataset"
    checker: Callable[..., dict[str, Any]] = None  # required in practice
    checker_kwargs: dict[str, Any] = field(default_factory=dict)


SoftSpec = SoftRowSpec | SoftDatasetSpec