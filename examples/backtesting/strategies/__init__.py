"""Strategy-specific example builders and end-to-end scripts."""

from .skew_mispricing import build_skew_strategy
from .vrp import build_vrp_strategy

__all__ = ["build_skew_strategy", "build_vrp_strategy"]
