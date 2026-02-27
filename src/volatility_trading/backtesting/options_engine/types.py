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
from typing import Any, Literal, TypeAlias

import pandas as pd

from volatility_trading.options import MarketState, OptionType, PositionSide

FillPolicy: TypeAlias = Literal["all_or_none", "min_ratio"]


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """Typed quote snapshot used by options-engine core lifecycle logic.

    This object intentionally captures the subset of fields required by
    entry/sizing/valuation paths so core modules do not depend on raw row keys.
    """

    option_type_label: str
    strike: float
    bid_price: float
    ask_price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    expiry_date: pd.Timestamp | None = None
    dte: int | None = None
    spot_price: float | None = None
    smoothed_iv: float | None = None
    yte: float | None = None
    open_interest: float | None = None
    volume: float | None = None

    @classmethod
    def from_series(cls, quote: pd.Series) -> QuoteSnapshot:
        """Build a typed snapshot from one chain row.

        Required fields:
            - ``option_type``
            - ``strike``

        Optional numeric fields (prices and Greeks) default to ``0.0`` when
        absent so historical tests and lightweight adapters can still pass
        sparse rows without coupling to full chain schemas.
        """
        option_type_raw = quote.get("option_type")
        if option_type_raw is None:
            raise KeyError("option_type")
        return cls(
            option_type_label=str(option_type_raw),
            strike=_required_float(quote.get("strike"), field="strike"),
            bid_price=_float_or_default(quote.get("bid_price"), default=0.0),
            ask_price=_float_or_default(quote.get("ask_price"), default=0.0),
            delta=_float_or_default(quote.get("delta"), default=0.0),
            gamma=_float_or_default(quote.get("gamma"), default=0.0),
            vega=_float_or_default(quote.get("vega"), default=0.0),
            theta=_float_or_default(quote.get("theta"), default=0.0),
            expiry_date=_optional_timestamp(quote.get("expiry_date")),
            dte=_optional_int(quote.get("dte")),
            spot_price=_optional_float(quote.get("spot_price")),
            smoothed_iv=_optional_float(quote.get("smoothed_iv")),
            yte=_optional_float(quote.get("yte")),
            open_interest=_optional_float(quote.get("open_interest")),
            volume=_optional_float(quote.get("volume")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to a plain mapping."""
        return {
            "option_type": self.option_type_label,
            "strike": self.strike,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "expiry_date": self.expiry_date,
            "dte": self.dte,
            "spot_price": self.spot_price,
            "smoothed_iv": self.smoothed_iv,
            "yte": self.yte,
            "open_interest": self.open_interest,
            "volume": self.volume,
        }


def _required_float(value: object, *, field: str) -> float:
    if value is None or pd.isna(value):
        raise KeyError(field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _float_or_default(value: object, *, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


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


@dataclass(frozen=True)
class LegSelection:
    """Concrete quote selected for one leg at entry.

    Attributes:
        spec: Original leg template used to select the quote.
        quote: Typed quote snapshot selected at entry.
        side: Trade side resolved to `PositionSide`.
        entry_price: Executed entry price for that leg after slippage.
    """

    spec: LegSpec
    quote: QuoteSnapshot
    side: PositionSide | int
    entry_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.quote, QuoteSnapshot):
            raise ValueError("quote must be QuoteSnapshot")

        if isinstance(self.side, PositionSide):
            return
        try:
            normalized = PositionSide(int(self.side))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "side must be PositionSide.SHORT (-1) or PositionSide.LONG (+1)"
            ) from exc
        object.__setattr__(self, "side", normalized)


@dataclass(frozen=True)
class EntryIntent:
    """Fully resolved structure entry consumed by sizing/lifecycle engines.

    Attributes:
        entry_date: Trade date when the structure is entered.
        expiry_date: Summary expiry used by reporting outputs.
        chosen_dte: Summary DTE used by reporting outputs.
        legs: Concrete selected legs with quotes and execution sides.
        entry_state: Market snapshot used for entry pricing/risk context.
    """

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    chosen_dte: int
    legs: tuple[LegSelection, ...]
    entry_state: MarketState | None = None

    def __post_init__(self) -> None:
        if self.chosen_dte <= 0:
            raise ValueError("chosen_dte must be > 0")
        if not self.legs:
            raise ValueError("legs must not be empty")
