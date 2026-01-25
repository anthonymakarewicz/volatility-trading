# qc/spec_types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import polars as pl


SoftKind = Literal["row", "dataset"]


@dataclass(frozen=True)
class SoftSpec:
    base_name: str
    kind: SoftKind = "row"
    flagger: Callable[..., pl.DataFrame] | None = None
    violation_col: str | None = None
    checker: Callable[..., dict[str, Any]] | None = None
    thresholds: dict[str, float] | None = None
    use_roi: bool = True
    by_option_type: bool = True
    requires_wide: bool = False
    flagger_kwargs: dict[str, Any] = field(default_factory=dict)
    checker_kwargs: dict[str, Any] = field(default_factory=dict)
    sample_cols: list[str] | None = None