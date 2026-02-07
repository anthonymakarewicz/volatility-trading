from __future__ import annotations

from .hard.specs import get_hard_specs
from .info.specs import get_info_specs
from .soft.specs import get_soft_specs

__all__ = [
    "get_hard_specs",
    "get_soft_specs",
    "get_info_specs",
]
