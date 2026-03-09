"""Shared execution economics helpers for options lifecycle and sizing."""

from .contracts.structures import LegSelection


def effective_leg_side(leg: LegSelection) -> int:
    """Return signed side after applying the leg ratio sign."""
    weight_sign = 1 if leg.spec.weight >= 0 else -1
    return int(leg.side) * weight_sign


def leg_units(leg: LegSelection) -> int:
    """Return absolute leg ratio units used for PnL/Greek aggregation."""
    return abs(int(leg.spec.weight))


def leg_contract_multiplier(
    leg: LegSelection, *, option_contract_multiplier: float
) -> float:
    """Return cash multiplier for one leg including contract multiplier and ratio."""
    return float(option_contract_multiplier * leg_units(leg))
