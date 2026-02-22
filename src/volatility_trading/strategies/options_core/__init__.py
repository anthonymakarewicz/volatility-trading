"""Shared strategy-side options contracts and adapter helpers."""

from .adapters import (
    normalize_chain_option_type,
    option_type_to_chain_label,
    quote_to_option_leg,
    quote_to_option_spec,
    time_to_expiry_years,
)
from .selectors import choose_expiry_by_target_dte, pick_quote_by_delta
from .sizing import (
    estimate_short_straddle_margin_per_contract,
    size_short_straddle_contracts,
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
    "pick_quote_by_delta",
    "choose_expiry_by_target_dte",
    "estimate_short_straddle_margin_per_contract",
    "size_short_straddle_contracts",
]
