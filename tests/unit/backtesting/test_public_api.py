import volatility_trading.backtesting.options_engine as options_engine
from volatility_trading.backtesting import (
    Backtester,
    BidAskFeeOptionExecutionModel,
    ColumnMapOptionsChainAdapter,
    DeltaHedgePolicy,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    StrategySpec,
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
    StrategySpec as EngineStrategySpec,
)


def test_backtesting_reexports_common_user_types() -> None:
    assert Backtester is EngineBacktester
    assert StrategySpec is EngineStrategySpec
    assert DeltaHedgePolicy is EngineDeltaHedgePolicy
    assert FixedDeltaBandModel is EngineFixedDeltaBandModel
    assert BidAskFeeOptionExecutionModel is EngineBidAskFeeOptionExecutionModel
    assert FixedBpsHedgeExecutionModel is EngineFixedBpsHedgeExecutionModel
    assert ColumnMapOptionsChainAdapter is EngineColumnMapOptionsChainAdapter


def test_options_engine_namespace_hides_runtime_internal_helpers() -> None:
    assert not hasattr(options_engine, "EntryIntent")
    assert not hasattr(options_engine, "SinglePositionHooks")
    assert not hasattr(options_engine, "apply_leg_liquidity_filters")
