"""Shared dataclasses for option stress-testing and risk sizing."""

from __future__ import annotations

from dataclasses import dataclass

from volatility_trading.options.types import MarketShock


@dataclass(frozen=True)
class StressScenario:
    """Named market shock used to stress option valuations."""

    name: str
    shock: MarketShock


@dataclass(frozen=True)
class StressPoint:
    """Scenario-level stressed PnL for one contract set."""

    scenario: StressScenario
    pnl: float


@dataclass(frozen=True)
class StressResult:
    """Aggregate stressed PnL output for risk sizing.

    `worst_loss` is stored as a non-negative magnitude.
    """

    worst_loss: float
    worst_scenario: StressScenario
    points: tuple[StressPoint, ...]

    def __post_init__(self) -> None:
        if self.worst_loss < 0:
            raise ValueError("worst_loss must be non-negative")
        if not self.points:
            raise ValueError("points must not be empty")
