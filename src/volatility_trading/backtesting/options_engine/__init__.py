"""Advanced public namespace for options-engine-specific backtesting helpers.

Most users should prefer imports from ``volatility_trading.backtesting``.
This namespace remains available for advanced options-engine configuration,
strategy specs, exit policies, execution models, and plan-building helpers.
"""

from .contracts.structures import LegSpec, StructureSpec
from .exit_rules import (
    ExitRule,
    ExitRuleSet,
    MaxHoldingExitRule,
    RebalanceExitRule,
    SameDayReentryPolicy,
    StopLossExitRule,
    TakeProfitExitRule,
)
from .lifecycle.hedge_execution import (
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    HedgeExecutionResult,
    MidNoCostHedgeExecutionModel,
)
from .lifecycle.option_execution import (
    BidAskFeeOptionExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionModel,
    OptionExecutionOrder,
    OptionExecutionResult,
)
from .plan_builder import build_options_execution_plan
from .specs import (
    DeltaBandModel,
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    LifecycleConfig,
    SizingPolicyConfig,
    StrategySpec,
    WWDeltaBandModel,
)

__all__ = [
    "LegSpec",
    "StructureSpec",
    "StrategySpec",
    "DeltaBandModel",
    "DeltaHedgePolicy",
    "FixedDeltaBandModel",
    "WWDeltaBandModel",
    "HedgeTriggerPolicy",
    "LifecycleConfig",
    "SizingPolicyConfig",
    "ExitRule",
    "RebalanceExitRule",
    "MaxHoldingExitRule",
    "StopLossExitRule",
    "TakeProfitExitRule",
    "ExitRuleSet",
    "SameDayReentryPolicy",
    "build_options_execution_plan",
    "HedgeExecutionResult",
    "HedgeExecutionModel",
    "MidNoCostHedgeExecutionModel",
    "FixedBpsHedgeExecutionModel",
    "OptionExecutionOrder",
    "OptionExecutionResult",
    "OptionExecutionModel",
    "MidNoCostOptionExecutionModel",
    "BidAskFeeOptionExecutionModel",
]
