"""Shared option-pricing dataclasses and aliases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal, TypeAlias


class OptionType(StrEnum):
    """Canonical option side labels used across pricing code."""

    CALL = "call"
    PUT = "put"


# Tolerant input type accepted at system boundaries (vendor data/tests).
OptionTypeInput: TypeAlias = OptionType | Literal["call", "put", "C", "P"]


@dataclass(frozen=True)
class OptionSpec:
    """Contract terms required for pricing one vanilla option."""

    strike: float
    time_to_expiry: float
    option_type: OptionTypeInput


@dataclass(frozen=True)
class MarketState:
    """Market inputs used by pricing engines."""

    spot: float
    volatility: float
    rate: float = 0.0
    dividend_yield: float = 0.0


@dataclass(frozen=True)
class PricingResult:
    """Option value and first/second-order sensitivities."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


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
class MarketShock:
    """Small perturbation around a reference market state.

    Units:
    - `d_spot`: currency units
    - `d_volatility`: volatility in decimals (0.01 = 1 vol point)
    - `d_rate`: rate shift in decimals
    - `dt_years`: forward elapsed time in years
    """

    d_spot: float = 0.0
    d_volatility: float = 0.0
    d_rate: float = 0.0
    dt_years: float = 0.0
