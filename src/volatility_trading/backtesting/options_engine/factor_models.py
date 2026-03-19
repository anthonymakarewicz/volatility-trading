"""Explicit factor-decomposition models for options-engine attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from volatility_trading.options import OptionType

from .contracts.market import QuoteSnapshot
from .contracts.structures import LegSelection, StructureSpec
from .economics import effective_leg_side, leg_units

FACTOR_VALUE_PREFIX = "factor_"
FACTOR_EXPOSURE_PREFIX = "factor_exposure_"
_ACRONYM_TOKENS = frozenset({"iv", "rr", "ts"})


@dataclass(frozen=True, slots=True)
class FactorSnapshot:
    """Resolved factor state and linear exposures for one position snapshot."""

    values: Mapping[str, float] = field(default_factory=dict)
    exposures: Mapping[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        """Return prefixed flat columns suitable for MTM/report tables."""
        flat: dict[str, float] = {}
        for name, value in self.values.items():
            flat[factor_value_column(str(name))] = float(value)
        for name, value in self.exposures.items():
            flat[factor_exposure_column(str(name))] = float(value)
        return flat

    def scaled_exposures(self, factor: float) -> FactorSnapshot:
        """Return the same factor state with exposures scaled by ``factor``."""
        return FactorSnapshot(
            values=dict(self.values),
            exposures={
                name: float(value) * float(factor)
                for name, value in self.exposures.items()
            },
        )

    def zero_exposures(self) -> FactorSnapshot:
        """Return the same factor state with all exposures reset to zero."""
        return FactorSnapshot(
            values=dict(self.values),
            exposures={name: 0.0 for name in self.exposures},
        )


class FactorDecompositionModel(Protocol):
    """Contract for explicit structure-aware PnL factor models."""

    def validate_structure(self, *, structure_spec: StructureSpec) -> None:
        """Raise when the model is incompatible with the configured structure."""

    def snapshot(
        self,
        *,
        legs: Sequence[LegSelection],
        leg_quotes: Sequence[QuoteSnapshot],
        option_contract_multiplier: float,
        contracts: int,
    ) -> FactorSnapshot:
        """Return factor state and exposures for one position snapshot."""


@dataclass(frozen=True, slots=True)
class RiskReversalFactorModel:
    """Same-expiry risk-reversal factor model using IV level and RR skew."""

    def validate_structure(self, *, structure_spec: StructureSpec) -> None:
        legs = structure_spec.legs
        if len(legs) != 2:
            raise ValueError("RiskReversalFactorModel requires exactly two legs")
        option_types = {leg.option_type for leg in legs}
        if option_types != {OptionType.CALL, OptionType.PUT}:
            raise ValueError(
                "RiskReversalFactorModel requires exactly one call leg and one put leg"
            )
        expiry_groups = {leg.expiry_group for leg in legs}
        if len(expiry_groups) != 1:
            raise ValueError(
                "RiskReversalFactorModel requires call and put legs to share one expiry_group"
            )

    def snapshot(
        self,
        *,
        legs: Sequence[LegSelection],
        leg_quotes: Sequence[QuoteSnapshot],
        option_contract_multiplier: float,
        contracts: int,
    ) -> FactorSnapshot:
        if len(legs) != len(leg_quotes):
            raise ValueError("legs and leg_quotes must have the same length")

        call_quote: QuoteSnapshot | None = None
        put_quote: QuoteSnapshot | None = None
        call_vega = 0.0
        put_vega = 0.0
        for leg, quote in zip(legs, leg_quotes, strict=True):
            signed_vega = (
                effective_leg_side(leg)
                * float(quote.vega)
                * float(leg_units(leg))
                * float(option_contract_multiplier)
                * float(contracts)
            )
            if leg.spec.option_type == OptionType.CALL:
                call_quote = quote
                call_vega += signed_vega
            elif leg.spec.option_type == OptionType.PUT:
                put_quote = quote
                put_vega += signed_vega
            else:
                raise ValueError("RiskReversalFactorModel only supports call/put legs")

        if call_quote is None or put_quote is None:
            raise ValueError(
                "RiskReversalFactorModel requires exactly one resolved call quote and one resolved put quote"
            )
        if call_quote.market_iv is None or put_quote.market_iv is None:
            raise ValueError(
                "RiskReversalFactorModel requires market_iv on both call and put quotes"
            )

        call_iv_pts = float(call_quote.market_iv) * 100.0
        put_iv_pts = float(put_quote.market_iv) * 100.0
        return FactorSnapshot(
            values={
                "iv_level": 0.5 * (call_iv_pts + put_iv_pts),
                "rr_skew": call_iv_pts - put_iv_pts,
            },
            exposures={
                "iv_level": call_vega + put_vega,
                "rr_skew": 0.5 * (call_vega - put_vega),
            },
        )


def factor_value_column(name: str) -> str:
    """Return the canonical MTM column name for one factor state value."""
    return f"{FACTOR_VALUE_PREFIX}{name}"


def factor_exposure_column(name: str) -> str:
    """Return the canonical MTM column name for one factor exposure."""
    return f"{FACTOR_EXPOSURE_PREFIX}{name}"


def factor_pnl_column(name: str) -> str:
    """Return the canonical attribution column name for one factor PnL."""
    display = "_".join(
        token.upper() if token in _ACRONYM_TOKENS else token.title()
        for token in name.split("_")
    )
    return f"{display}_PnL"


def factor_names_from_columns(columns: Sequence[str]) -> list[str]:
    """Extract known factor names from prefixed MTM/report columns."""
    names: set[str] = set()
    for column in columns:
        if column.startswith(FACTOR_EXPOSURE_PREFIX):
            names.add(column.removeprefix(FACTOR_EXPOSURE_PREFIX))
            continue
        if column.startswith(FACTOR_VALUE_PREFIX):
            names.add(column.removeprefix(FACTOR_VALUE_PREFIX))
    return sorted(names)


__all__ = [
    "FACTOR_EXPOSURE_PREFIX",
    "FACTOR_VALUE_PREFIX",
    "FactorDecompositionModel",
    "FactorSnapshot",
    "RiskReversalFactorModel",
    "factor_exposure_column",
    "factor_names_from_columns",
    "factor_pnl_column",
    "factor_value_column",
]
