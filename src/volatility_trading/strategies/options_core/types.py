"""Shared strategy-side contracts for building multi-leg option entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

from volatility_trading.options import OptionType

FillPolicy: TypeAlias = Literal["all_or_none", "min_ratio"]


@dataclass(frozen=True)
class LegSpec:
    """Leg selection constraints used by structure builders."""

    option_type: OptionType
    delta_target: float
    delta_tolerance: float = 0.10
    weight: int = 1
    min_open_interest: int = 0
    min_volume: int = 0
    max_relative_spread: float | None = None

    def __post_init__(self) -> None:
        if self.delta_tolerance <= 0:
            raise ValueError("delta_tolerance must be > 0")
        if self.weight == 0:
            raise ValueError("weight must be non-zero")
        if self.min_open_interest < 0:
            raise ValueError("min_open_interest must be >= 0")
        if self.min_volume < 0:
            raise ValueError("min_volume must be >= 0")
        if self.max_relative_spread is not None and self.max_relative_spread < 0:
            raise ValueError("max_relative_spread must be >= 0 when provided")


@dataclass(frozen=True)
class StructureSpec:
    """Target structure definition resolved into concrete legs at entry time."""

    name: str
    dte_target: int = 30  # TODO: maybe move them to LegSpec e.g. a calendar spread would u different DTE in its Structure
    dte_tolerance: int = 7
    legs: tuple[LegSpec, ...] = ()
    fill_policy: FillPolicy = "all_or_none"
    min_fill_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.dte_target <= 0:
            raise ValueError("dte_target must be > 0")
        if self.dte_tolerance < 0:
            raise ValueError("dte_tolerance must be >= 0")
        if not self.legs:
            raise ValueError("legs must not be empty")
        if not 0 < self.min_fill_ratio <= 1:
            raise ValueError("min_fill_ratio must be in (0, 1]")
        if self.fill_policy == "all_or_none" and self.min_fill_ratio != 1.0:
            raise ValueError("all_or_none requires min_fill_ratio=1.0")


# TODO: Since it is a rela contratc, why not using OptionLeg which has the same args and is not
# specific to strategy only it is global to pricing and risk too sicn ethey work all with
# a real option contract ?
@dataclass(frozen=True)
class LegSelection:
    """Concrete quote selected for one leg at entry."""

    spec: LegSpec
    quote: pd.Series
    side: int
    entry_price: float

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("side must be -1 (short) or +1 (long)")


@dataclass(frozen=True)
class EntryIntent:
    """Resolved entry payload consumed by shared lifecycle components."""

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    chosen_dte: int
    legs: tuple[LegSelection, ...]
    spot: float | None = None  # TODO: Why not a MarketState instead ?
    volatility: float | None = None

    def __post_init__(self) -> None:
        if self.chosen_dte <= 0:
            raise ValueError("chosen_dte must be > 0")
        if not self.legs:
            raise ValueError("legs must not be empty")
