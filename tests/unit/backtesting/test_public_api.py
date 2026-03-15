import volatility_trading.backtesting as backtesting
import volatility_trading.backtesting.options_engine as options_engine
from volatility_trading.backtesting import (
    Backtester,
    BidAskFeeOptionExecutionModel,
    ColumnMapOptionsChainAdapter,
    DeltaHedgePolicy,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    StopLossExitRule,
    StrategySpec,
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


def test_backtesting_reexports_common_user_types() -> None:
    assert Backtester is EngineBacktester
    assert StrategySpec is EngineStrategySpec
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


def test_options_engine_namespace_hides_runtime_internal_helpers() -> None:
    assert not hasattr(options_engine, "EntryIntent")
    assert not hasattr(options_engine, "SinglePositionHooks")
    assert not hasattr(options_engine, "apply_leg_liquidity_filters")
    assert not hasattr(options_engine, "validate_options_chain")
    assert not hasattr(options_engine, "normalize_and_validate_options_chain")
    assert not hasattr(options_engine, "validate_options_chain_contract")
    assert not hasattr(options_engine, "ValidationMode")


def test_root_backtesting_namespace_hides_validation_mode() -> None:
    assert not hasattr(backtesting, "ValidationMode")
