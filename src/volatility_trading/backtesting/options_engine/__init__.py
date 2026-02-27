"""Shared options backtesting runtime contracts and helpers."""

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
    MtmRecord,
    OpenPosition,
    PositionEntrySetup,
    PositionLifecycleEngine,
)
from .selectors import (
    apply_leg_liquidity_filters,
    score_leg_candidates,
    select_best_expiry_for_leg_group,
    select_best_quote_for_leg,
)
from .sizing import (
    SizingDecision,
    SizingRequest,
    estimate_entry_intent_margin_per_contract,
    size_entry_intent,
    size_entry_intent_contracts,
)
from .specs import StrategySpec
from .types import EntryIntent, LegSelection, LegSpec, QuoteSnapshot, StructureSpec

__all__ = [
    "LegSpec",
    "QuoteSnapshot",
    "StructureSpec",
    "LegSelection",
    "EntryIntent",
    "normalize_chain_option_type",
    "option_type_to_chain_label",
    "time_to_expiry_years",
    "quote_to_option_spec",
    "quote_to_option_leg",
    "StrategySpec",
    "chain_for_date",
    "normalize_signals_to_on",
    "build_entry_intent_from_structure",
    "ExitRule",
    "RebalanceExitRule",
    "MaxHoldingExitRule",
    "ExitRuleSet",
    "SameDayReentryPolicy",
    "apply_leg_liquidity_filters",
    "score_leg_candidates",
    "select_best_quote_for_leg",
    "select_best_expiry_for_leg_group",
    "estimate_entry_intent_margin_per_contract",
    "SizingRequest",
    "SizingDecision",
    "size_entry_intent",
    "size_entry_intent_contracts",
    "PositionEntrySetup",
    "MtmRecord",
    "OpenPosition",
    "PositionLifecycleEngine",
]
