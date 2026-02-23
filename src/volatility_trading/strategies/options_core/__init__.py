"""Shared strategy-side options contracts and adapter helpers."""

from .adapters import (
    normalize_chain_option_type,
    option_type_to_chain_label,
    quote_to_option_leg,
    quote_to_option_spec,
    time_to_expiry_years,
)
from .entry import (
    build_entry_intent_from_structure,
    chain_for_date,
    normalize_signals_to_on,
)
from .exit_rules import (
    ExitRule,
    ExitRuleSet,
    MaxHoldingExitRule,
    RebalanceExitRule,
    SameDayReentryPolicy,
)
from .lifecycle import (
    OpenPosition,
    PositionEntrySetup,
    PositionLifecycleEngine,
)
from .runner import SinglePositionRunnerHooks, run_single_position_date_loop
from .selectors import choose_expiry_by_target_dte, pick_quote_by_delta
from .sizing import (
    estimate_entry_intent_margin_per_contract,
    estimate_short_straddle_margin_per_contract,
    estimate_structure_margin_per_contract,
    size_entry_intent_contracts,
    size_short_straddle_contracts,
    size_structure_contracts,
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
    "chain_for_date",
    "normalize_signals_to_on",
    "build_entry_intent_from_structure",
    "ExitRule",
    "RebalanceExitRule",
    "MaxHoldingExitRule",
    "ExitRuleSet",
    "SameDayReentryPolicy",
    "pick_quote_by_delta",
    "choose_expiry_by_target_dte",
    "estimate_short_straddle_margin_per_contract",
    "size_short_straddle_contracts",
    "estimate_structure_margin_per_contract",
    "size_structure_contracts",
    "estimate_entry_intent_margin_per_contract",
    "size_entry_intent_contracts",
    "PositionEntrySetup",
    "OpenPosition",
    "PositionLifecycleEngine",
    "SinglePositionRunnerHooks",
    "run_single_position_date_loop",
]
