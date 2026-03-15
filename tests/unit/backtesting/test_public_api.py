import volatility_trading.backtesting as backtesting
import volatility_trading.backtesting.data_adapters as data_adapters
import volatility_trading.backtesting.options_engine as options_engine
import volatility_trading.backtesting.performance as performance
import volatility_trading.backtesting.reporting as reporting
from volatility_trading.backtesting import (
    Backtester,
    BidAskFeeOptionExecutionModel,
    ColumnMapOptionsChainAdapter,
    DeltaHedgePolicy,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    StopLossExitRule,
    TakeProfitExitRule,
    canonicalize_options_chain_for_backtest,
    filter_options_chain_for_backtest,
    load_daily_features_frame,
    load_fred_rate_series,
    load_orats_options_chain_for_backtest,
    load_yfinance_close_series,
    spot_series_from_options_chain,
)
from volatility_trading.backtesting.engine import Backtester as EngineBacktester
from volatility_trading.backtesting.options_engine import (
    AliasOptionsChainAdapter as EngineAliasOptionsChainAdapter,
)
from volatility_trading.backtesting.options_engine import (
    BidAskFeeOptionExecutionModel as EngineBidAskFeeOptionExecutionModel,
)
from volatility_trading.backtesting.options_engine import (
    ColumnMapOptionsChainAdapter as EngineColumnMapOptionsChainAdapter,
)
from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy as EngineDeltaHedgePolicy,
)
from volatility_trading.backtesting.options_engine import (
    FixedBpsHedgeExecutionModel as EngineFixedBpsHedgeExecutionModel,
)
from volatility_trading.backtesting.options_engine import (
    FixedDeltaBandModel as EngineFixedDeltaBandModel,
)
from volatility_trading.backtesting.options_engine import (
    StopLossExitRule as EngineStopLossExitRule,
)
from volatility_trading.backtesting.options_engine import (
    StrategySpec as EngineStrategySpec,
)
from volatility_trading.backtesting.options_engine import (
    TakeProfitExitRule as EngineTakeProfitExitRule,
)
from volatility_trading.backtesting.performance import (
    PerformanceMetricsBundle as PerformanceBundle,
)
from volatility_trading.backtesting.reporting import (
    BacktestReportBundle as ReportBundle,
)


def test_backtesting_reexports_common_user_types() -> None:
    assert Backtester is EngineBacktester
    assert DeltaHedgePolicy is EngineDeltaHedgePolicy
    assert FixedDeltaBandModel is EngineFixedDeltaBandModel
    assert StopLossExitRule is EngineStopLossExitRule
    assert TakeProfitExitRule is EngineTakeProfitExitRule
    assert BidAskFeeOptionExecutionModel is EngineBidAskFeeOptionExecutionModel
    assert FixedBpsHedgeExecutionModel is EngineFixedBpsHedgeExecutionModel
    assert ColumnMapOptionsChainAdapter is EngineColumnMapOptionsChainAdapter
    assert callable(canonicalize_options_chain_for_backtest)
    assert callable(filter_options_chain_for_backtest)
    assert callable(load_daily_features_frame)
    assert callable(load_orats_options_chain_for_backtest)
    assert callable(load_fred_rate_series)
    assert callable(load_yfinance_close_series)
    assert callable(spot_series_from_options_chain)
    assert callable(backtesting.compute_performance_metrics)
    assert callable(backtesting.format_performance_report)
    assert callable(backtesting.print_performance_report)


def test_options_engine_namespace_hides_runtime_internal_helpers() -> None:
    assert not hasattr(options_engine, "EntryIntent")
    assert not hasattr(options_engine, "SinglePositionHooks")
    assert not hasattr(options_engine, "apply_leg_liquidity_filters")
    assert not hasattr(options_engine, "validate_options_chain")
    assert not hasattr(options_engine, "normalize_and_validate_options_chain")
    assert not hasattr(options_engine, "validate_options_chain_contract")
    assert not hasattr(options_engine, "ValidationMode")
    assert not hasattr(options_engine, "coerce_options_frame_to_pandas")
    assert not hasattr(options_engine, "normalize_options_chain")


def test_root_backtesting_namespace_hides_validation_mode() -> None:
    assert not hasattr(backtesting, "ValidationMode")


def test_root_backtesting_namespace_hides_advanced_spec_and_plumbing_types() -> None:
    assert not hasattr(backtesting, "OptionsChainAdapter")
    assert not hasattr(backtesting, "AliasOptionsChainAdapter")
    assert not hasattr(backtesting, "StrategySpec")
    assert not hasattr(backtesting, "StructureSpec")
    assert not hasattr(backtesting, "LegSpec")
    assert not hasattr(backtesting, "LifecycleConfig")
    assert not hasattr(backtesting, "SizingPolicyConfig")
    assert not hasattr(backtesting, "MarginStatus")
    assert not hasattr(backtesting, "MarginCore")
    assert not hasattr(backtesting, "HedgeExecutionModel")
    assert not hasattr(backtesting, "OptionExecutionModel")
    assert not hasattr(backtesting, "ConstantRateModel")
    assert not hasattr(backtesting, "SeriesRateModel")
    assert not hasattr(backtesting, "summarize_by_contracts")
    assert not hasattr(backtesting, "format_stressed_risk_report")
    assert not hasattr(backtesting, "print_stressed_risk_metrics")


def test_options_engine_namespace_retains_advanced_spec_and_adapter_types() -> None:
    assert EngineStrategySpec is options_engine.StrategySpec
    assert EngineAliasOptionsChainAdapter is options_engine.AliasOptionsChainAdapter


def test_performance_namespace_retains_advanced_public_bundle_types() -> None:
    assert performance.PerformanceMetricsBundle is PerformanceBundle
    assert callable(performance.compute_performance_metrics)
    assert callable(performance.summarize_by_contracts)


def test_reporting_namespace_retains_advanced_public_bundle_types() -> None:
    assert reporting.BacktestReportBundle is ReportBundle
    assert callable(reporting.build_backtest_report_bundle)
    assert callable(reporting.save_backtest_report_bundle)
    assert callable(reporting.plot_performance_dashboard)


def test_data_adapters_namespace_is_not_curated_public_facade() -> None:
    assert data_adapters.__all__ == []
    assert not hasattr(data_adapters, "ColumnMapOptionsChainAdapter")
    assert not hasattr(data_adapters, "validate_options_chain")
