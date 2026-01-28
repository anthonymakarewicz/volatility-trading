# qc/info/specs_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class InfoSpec:
    base_name: str
    summarizer: Callable[..., dict[str, Any]]
    summarizer_kwargs: dict[str, Any] | None = None