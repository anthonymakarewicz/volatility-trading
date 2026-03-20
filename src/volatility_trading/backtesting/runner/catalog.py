"""Shared internal registries/constants for runner-supported components."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from volatility_trading.backtesting.options_engine.lifecycle import (
    BidAskFeeOptionExecutionModel,
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionModel,
)
from volatility_trading.options import (
    FixedGridScenarioGenerator,
    MarginModel,
    PortfolioMarginProxyModel,
    RegTMarginModel,
    ScenarioGenerator,
)

OptionExecutionFactory: TypeAlias = Callable[..., OptionExecutionModel]
HedgeExecutionFactory: TypeAlias = Callable[..., HedgeExecutionModel]
MarginModelFactory: TypeAlias = Callable[..., MarginModel]
ScenarioGeneratorFactory: TypeAlias = Callable[..., ScenarioGenerator]

OPTION_EXECUTION_MODEL_FACTORIES: dict[str, OptionExecutionFactory] = {
    "bid_ask_fee": BidAskFeeOptionExecutionModel,
    "mid_no_cost": MidNoCostOptionExecutionModel,
}
HEDGE_EXECUTION_MODEL_FACTORIES: dict[str, HedgeExecutionFactory] = {
    "fixed_bps": FixedBpsHedgeExecutionModel,
    "mid_no_cost": MidNoCostHedgeExecutionModel,
}
MARGIN_MODEL_FACTORIES: dict[str, MarginModelFactory] = {
    "portfolio_margin_proxy": PortfolioMarginProxyModel,
    "regt": RegTMarginModel,
}
SCENARIO_GENERATOR_FACTORIES: dict[str, ScenarioGeneratorFactory] = {
    "fixed_grid": FixedGridScenarioGenerator,
}
