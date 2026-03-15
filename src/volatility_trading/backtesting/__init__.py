"""Preferred public import surface for common backtesting workflows.

Most users should start from this package root for runtime setup, canonical
data loading, common hedging/exit configuration, and standard performance
reporting. More specialized helpers remain in narrower advanced subpackages
such as ``volatility_trading.backtesting.options_engine``,
``volatility_trading.backtesting.performance``, and
``volatility_trading.backtesting.reporting``.
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
from .data_adapters.options_chain_adapters import (
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    YfinanceOptionsChainAdapter,
)
from .data_adapters.options_chain_pipeline import OptionsChainAdapterError
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
