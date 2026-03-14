"""Advanced public namespace for options-engine-specific backtesting helpers.

Most users should prefer imports from ``volatility_trading.backtesting``.
This namespace remains available for advanced options-engine configuration and
extension points.
"""

from ..data_adapters import (
    CANONICAL_OPTIONAL_COLUMNS,
    CANONICAL_REQUIRED_COLUMNS,
    AliasOptionsChainAdapter,
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    OptionsChainAdapter,
    OptionsChainAdapterError,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    ValidationMode,
    YfinanceOptionsChainAdapter,
    coerce_options_frame_to_pandas,
    normalize_and_validate_options_chain,
    normalize_options_chain,
    validate_options_chain,
    validate_options_chain_contract,
)
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
    "OptionsChainAdapter",
    "OptionsChainAdapterError",
    "AliasOptionsChainAdapter",
    "CanonicalOptionsChainAdapter",
    "OratsOptionsChainAdapter",
    "YfinanceOptionsChainAdapter",
    "ColumnMapOptionsChainAdapter",
    "OptionsDxOptionsChainAdapter",
    "coerce_options_frame_to_pandas",
    "normalize_and_validate_options_chain",
    "normalize_options_chain",
    "validate_options_chain",
    "validate_options_chain_contract",
    "ValidationMode",
    "CANONICAL_REQUIRED_COLUMNS",
    "CANONICAL_OPTIONAL_COLUMNS",
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
