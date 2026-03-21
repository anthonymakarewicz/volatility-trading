"""Shared dataclasses for option stress-testing and risk sizing."""

from __future__ import annotations

from collections.abc import Sequence
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

    @classmethod
    def from_points(cls, points: Sequence[StressPoint]) -> StressResult:
        """Build a stress summary from scenario-level PnL points.

        Args:
            points: Scenario-level stressed PnL observations.

        Returns:
            Stress summary with `worst_loss` derived from the most negative PnL
            point and stored as a non-negative magnitude.

        Raises:
            ValueError: If `points` is empty.
        """
        normalized_points = tuple(points)
        if not normalized_points:
            raise ValueError("points must not be empty")
        worst_point = min(normalized_points, key=lambda point: point.pnl)
        return cls(
            worst_loss=max(-worst_point.pnl, 0.0),
            worst_scenario=worst_point.scenario,
            points=normalized_points,
        )

    def __post_init__(self) -> None:
        if self.worst_loss < 0:
            raise ValueError("worst_loss must be non-negative")
        if not self.points:
            raise ValueError("points must not be empty")
