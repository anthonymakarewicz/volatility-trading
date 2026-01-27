from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class HardSpec:
    """Definition of one HARD (must-pass) row-level QC check."""
    name: str
    predicate_expr: pl.Expr
    sample_cols: Sequence[str] | None = None
    sample_n: int = 10