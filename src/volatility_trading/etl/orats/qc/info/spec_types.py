# qc/info/specs_types.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InfoSpec:
    base_name: str
    summarizer: Callable[..., dict[str, Any]]
    summarizer_kwargs: dict[str, Any] | None = None
