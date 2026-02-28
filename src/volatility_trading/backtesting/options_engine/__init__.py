"""Shared options backtesting runtime contracts and helpers."""

from .adapters import (
    normalize_chain_option_type,
    option_type_to_chain_label,
    quote_to_option_leg,
    quote_to_option_spec,
    time_to_expiry_years,
)
from .contracts import SinglePositionExecutionPlan, SinglePositionHooks
from .contracts.market import QuoteSnapshot
from .contracts.records import MtmRecord, TradeRecord
from .contracts.runtime import (
    HedgeState,
    LifecycleStepResult,
    OpenPosition,
    PositionEntrySetup,
)
from .contracts.structures import EntryIntent, LegSelection, LegSpec, StructureSpec
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
from .lifecycle import PositionLifecycleEngine
from .outputs import build_options_backtest_outputs
from .plan_builder import build_options_execution_plan
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
)
from .specs import (
    DeltaHedgePolicy,
    HedgeTriggerPolicy,
    LifecycleConfig,
    SizingPolicyConfig,
    StrategySpec,
)

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
    "DeltaHedgePolicy",
    "HedgeTriggerPolicy",
    "LifecycleConfig",
    "SizingPolicyConfig",
    "SinglePositionHooks",
    "SinglePositionExecutionPlan",
    "build_options_execution_plan",
    "build_options_backtest_outputs",
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
    "PositionEntrySetup",
    "MtmRecord",
    "TradeRecord",
    "OpenPosition",
    "HedgeState",
    "LifecycleStepResult",
    "PositionLifecycleEngine",
]
