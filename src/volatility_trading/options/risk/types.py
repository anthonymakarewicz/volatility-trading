"""Shared dataclasses for option stress-testing and risk sizing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from volatility_trading.options.types import MarketShock, OptionSpec


class PositionSide(IntEnum):
    """Signed position direction used for PnL aggregation."""

    SHORT = -1
    LONG = 1


@dataclass(frozen=True)
class OptionLeg:
    """One option leg with entry economics for stress revaluation.

    `contract_multiplier` is the lot-size cash scalar (e.g., 100 for US equity
    options).
    """

    spec: OptionSpec
    entry_price: float
    side: PositionSide
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.side, PositionSide):
            raise ValueError("side must be PositionSide.SHORT or PositionSide.LONG")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be > 0")


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
