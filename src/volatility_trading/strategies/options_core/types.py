"""Core data contracts for options strategy entry selection and lifecycle.

These dataclasses define the handoff between:
- structure selection (`LegSpec` / `StructureSpec`),
- selected chain rows (`LegSelection`), and
- lifecycle execution (`EntryIntent`).

The design intentionally separates abstract leg constraints from concrete quote
snapshots so strategy builders and lifecycle accounting can evolve independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

from volatility_trading.options import OptionType

FillPolicy: TypeAlias = Literal["all_or_none", "min_ratio"]


@dataclass(frozen=True)
class LegSpec:
    """Selection constraints for one logical leg in a structure template.

    Attributes:
        option_type: Call or put side to search in the chain.
        delta_target: Target option delta used for moneyness selection.
        delta_tolerance: Allowed absolute deviation around ``delta_target``.
        expiry_group: Group key used to force a shared expiry across related legs.
        dte_target: Optional group-level DTE override for this leg's expiry group.
        dte_tolerance: Optional DTE tolerance override for this leg's expiry group.
        weight: Ratio multiplier for this leg (for example +1/-2/+1 butterfly).
        min_open_interest: Hard lower bound on open interest for eligible quotes.
        min_volume: Hard lower bound on volume for eligible quotes.
        max_relative_spread: Optional hard cap on relative bid/ask spread.
    """

    option_type: OptionType
    delta_target: float
    delta_tolerance: float = 0.10
    expiry_group: str = "main"
    dte_target: int | None = None
    dte_tolerance: int | None = None
    weight: int = 1
    min_open_interest: int = 0
    min_volume: int = 0
    max_relative_spread: float | None = None

    def __post_init__(self) -> None:
        if self.delta_tolerance <= 0:
            raise ValueError("delta_tolerance must be > 0")
        if not self.expiry_group:
            raise ValueError("expiry_group must be non-empty")
        if self.dte_target is not None and self.dte_target <= 0:
            raise ValueError("dte_target must be > 0 when provided")
        if self.dte_tolerance is not None and self.dte_tolerance < 0:
            raise ValueError("dte_tolerance must be >= 0 when provided")
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
    """Template describing the target multi-leg structure to open.

    Attributes:
        name: Human-readable structure identifier for reporting/debugging.
        dte_target: Default target DTE applied when legs do not override it.
        dte_tolerance: Default DTE tolerance applied when legs do not override it.
        legs: Ordered tuple of leg templates to resolve in the chain.
        fill_policy: Entry completeness rule (strict all legs vs minimum ratio).
        min_fill_ratio: Minimum selected-leg ratio when ``fill_policy='min_ratio'``.
    """

    name: str
    dte_target: int = 30
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
    """Concrete quote selected for one leg at entry.

    Attributes:
        spec: Original leg template used to select the quote.
        quote: Snapshot row from the chain at entry.
        side: Trade side (+1 long, -1 short) resolved from strategy direction.
        entry_price: Executed entry price for that leg after slippage.
    """

    spec: LegSpec
    quote: pd.Series
    side: int
    entry_price: float

    def __post_init__(self) -> None:
        if self.side not in (-1, 1):
            raise ValueError("side must be -1 (short) or +1 (long)")


@dataclass(frozen=True)
class EntryIntent:
    """Fully resolved structure entry consumed by sizing/lifecycle engines.

    Attributes:
        entry_date: Trade date when the structure is entered.
        expiry_date: Summary expiry used by legacy reporting fields.
        chosen_dte: Summary DTE used by legacy reporting fields.
        legs: Concrete selected legs with quotes and execution sides.
        spot: Spot level used for entry pricing/risk context.
        volatility: Volatility proxy used for pricing/risk context.
    """

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    chosen_dte: int
    legs: tuple[LegSelection, ...]
    spot: float | None = None
    volatility: float | None = None

    def __post_init__(self) -> None:
        if self.chosen_dte <= 0:
            raise ValueError("chosen_dte must be > 0")
        if not self.legs:
            raise ValueError("legs must not be empty")
