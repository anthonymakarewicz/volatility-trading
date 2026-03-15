"""Preferred public import surface for backtesting users.

Most user-facing backtesting types should be imported from this package root.
Engine-specific helpers remain available under
``volatility_trading.backtesting.options_engine`` for advanced use.
"""

from .attribution import to_daily_mtm
from .config import (
    AccountConfig,
    BacktestRunConfig,
    BrokerConfig,
    ExecutionConfig,
    MarginConfig,
    ModelingConfig,
)
from .data_adapters import (
    AliasOptionsChainAdapter,
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    OptionsChainAdapter,
    OptionsChainAdapterError,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    YfinanceOptionsChainAdapter,
)
from .data_contracts import (
    HedgeMarketData,
    OptionsBacktestDataBundle,
    OptionsMarketData,
)
from .data_loading import (
    canonicalize_options_chain_for_backtest,
    filter_options_chain_for_backtest,
    load_daily_features_frame,
    load_fred_rate_series,
    load_orats_options_chain_for_backtest,
    load_yfinance_close_series,
    spot_series_from_options_chain,
)
from .engine import Backtester
from .margin import MarginAccount, MarginPolicy, MarginStatus
from .margin_types import MarginCore
from .options_engine import (
    BidAskFeeOptionExecutionModel,
    DeltaHedgePolicy,
    ExitRuleSet,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    HedgeExecutionModel,
    HedgeTriggerPolicy,
    LegSpec,
    LifecycleConfig,
    MaxHoldingExitRule,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionModel,
    RebalanceExitRule,
    SameDayReentryPolicy,
    SizingPolicyConfig,
    StopLossExitRule,
    StrategySpec,
    StructureSpec,
    TakeProfitExitRule,
    WWDeltaBandModel,
)
from .performance import (
    compute_performance_metrics,
    format_performance_report,
    format_stressed_risk_report,
    print_performance_report,
    print_stressed_risk_metrics,
    summarize_by_contracts,
)
from .rates import (
    ConstantRateModel,
    RateModel,
    SeriesRateModel,
    coerce_rate_model,
)
from .reporting import (
    build_backtest_report_bundle,
    save_backtest_report_bundle,
)
from .reporting.plots import (
    plot_pnl_attribution,
    plot_stressed_pnl,
)

__all__ = [
    "AccountConfig",
    "ExecutionConfig",
    "MarginConfig",
    "BrokerConfig",
    "ModelingConfig",
    "BacktestRunConfig",
    "Backtester",
    "OptionsBacktestDataBundle",
    "OptionsMarketData",
    "HedgeMarketData",
    "canonicalize_options_chain_for_backtest",
    "filter_options_chain_for_backtest",
    "load_daily_features_frame",
    "load_orats_options_chain_for_backtest",
    "load_fred_rate_series",
    "load_yfinance_close_series",
    "spot_series_from_options_chain",
    "OptionsChainAdapter",
    "OptionsChainAdapterError",
    "AliasOptionsChainAdapter",
    "CanonicalOptionsChainAdapter",
    "OratsOptionsChainAdapter",
    "YfinanceOptionsChainAdapter",
    "ColumnMapOptionsChainAdapter",
    "OptionsDxOptionsChainAdapter",
    "StrategySpec",
    "StructureSpec",
    "LegSpec",
    "LifecycleConfig",
    "SizingPolicyConfig",
    "ExitRuleSet",
    "RebalanceExitRule",
    "MaxHoldingExitRule",
    "StopLossExitRule",
    "TakeProfitExitRule",
    "SameDayReentryPolicy",
    "DeltaHedgePolicy",
    "HedgeTriggerPolicy",
    "FixedDeltaBandModel",
    "WWDeltaBandModel",
    "HedgeExecutionModel",
    "MidNoCostHedgeExecutionModel",
    "FixedBpsHedgeExecutionModel",
    "OptionExecutionModel",
    "MidNoCostOptionExecutionModel",
    "BidAskFeeOptionExecutionModel",
    "MarginCore",
    "to_daily_mtm",
    "MarginPolicy",
    "MarginStatus",
    "MarginAccount",
    "RateModel",
    "ConstantRateModel",
    "SeriesRateModel",
    "coerce_rate_model",
    "compute_performance_metrics",
    "summarize_by_contracts",
    "format_performance_report",
    "format_stressed_risk_report",
    "print_performance_report",
    "print_stressed_risk_metrics",
    "plot_pnl_attribution",
    "plot_stressed_pnl",
    "build_backtest_report_bundle",
    "save_backtest_report_bundle",
]
