"""Shared strategy runtime contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from volatility_trading.backtesting.types import SliceContext


class Strategy(ABC):
    """Abstract strategy runtime used by the backtesting engine."""

    @abstractmethod
    def run(self, ctx: SliceContext) -> tuple[Any, Any]:
        """Execute strategy and return `(trades, mtm)`."""
