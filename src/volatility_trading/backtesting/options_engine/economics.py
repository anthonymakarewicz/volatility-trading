"""Shared execution economics helpers for options lifecycle and sizing."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts.structures import LegSelection


def effective_leg_side(leg: LegSelection) -> int:
    """Return signed side after applying the leg ratio sign."""
    weight_sign = 1 if leg.spec.weight >= 0 else -1
    return int(leg.side) * weight_sign


def leg_units(leg: LegSelection) -> int:
    """Return absolute leg ratio units used for PnL/Greek aggregation."""
    return abs(int(leg.spec.weight))


def leg_contract_multiplier(leg: LegSelection, *, lot_size: int) -> float:
    """Return cash multiplier for one leg including lot size and ratio units."""
    return float(lot_size * leg_units(leg))


def roundtrip_commission_per_structure_contract(
    *,
    commission_per_leg: float,
    legs: Sequence[LegSelection],
) -> float:
    """Return roundtrip commission for one opened structure contract."""
    return 2.0 * float(commission_per_leg) * float(len(legs))
