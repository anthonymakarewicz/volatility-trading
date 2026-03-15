"""Preferred public import surface for common backtesting entrypoints.

Most users should start from this package root for runtime setup, data loading,
and common hedging/exit configuration. Advanced strategy/spec construction
helpers remain available under ``volatility_trading.backtesting.options_engine``.
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
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
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
from .margin import MarginAccount, MarginPolicy
from .options_engine import (
    BidAskFeeOptionExecutionModel,
    DeltaHedgePolicy,
    ExitRuleSet,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    MaxHoldingExitRule,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    RebalanceExitRule,
    SameDayReentryPolicy,
    StopLossExitRule,
    TakeProfitExitRule,
    WWDeltaBandModel,
)
from .performance import (
    compute_performance_metrics,
    format_performance_report,
    print_performance_report,
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
    "OptionsChainAdapterError",
    "CanonicalOptionsChainAdapter",
    "OratsOptionsChainAdapter",
    "YfinanceOptionsChainAdapter",
    "ColumnMapOptionsChainAdapter",
    "OptionsDxOptionsChainAdapter",
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
    "MidNoCostHedgeExecutionModel",
    "FixedBpsHedgeExecutionModel",
    "MidNoCostOptionExecutionModel",
    "BidAskFeeOptionExecutionModel",
    "to_daily_mtm",
    "MarginPolicy",
    "MarginAccount",
    "compute_performance_metrics",
    "format_performance_report",
    "print_performance_report",
]
