"""Spec dataclasses for INFO QC checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InfoSpec:
    """Definition of one informational QC summarizer."""

    base_name: str
    summarizer: Callable[..., dict[str, Any]]
    summarizer_kwargs: dict[str, Any] | None = None
