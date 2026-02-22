"""Shared strategy-side options contracts and adapter helpers."""

from .adapters import (
    normalize_chain_option_type,
    option_type_to_chain_label,
    quote_to_option_leg,
    quote_to_option_spec,
    time_to_expiry_years,
)
from .types import EntryIntent, LegSelection, LegSpec, StructureSpec

__all__ = [
    "LegSpec",
    "StructureSpec",
    "LegSelection",
    "EntryIntent",
    "normalize_chain_option_type",
    "option_type_to_chain_label",
    "time_to_expiry_years",
    "quote_to_option_spec",
    "quote_to_option_leg",
]
