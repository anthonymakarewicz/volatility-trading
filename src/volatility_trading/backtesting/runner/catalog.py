"""Shared internal registries/constants for runner-supported components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from volatility_trading.backtesting.data_adapters import (
    CanonicalOptionsChainAdapter,
    OptionsChainAdapter,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    YfinanceOptionsChainAdapter,
)
from volatility_trading.backtesting.options_engine.lifecycle import (
    BidAskFeeOptionExecutionModel,
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionModel,
)
from volatility_trading.options import (
    MarginModel,
    PortfolioMarginProxyModel,
    RegTMarginModel,
)

OptionExecutionFactory: TypeAlias = Callable[..., OptionExecutionModel]
HedgeExecutionFactory: TypeAlias = Callable[..., HedgeExecutionModel]
MarginModelFactory: TypeAlias = Callable[..., MarginModel]
OptionsAdapterFactory: TypeAlias = Callable[[], OptionsChainAdapter]

OPTIONS_SOURCE_PROVIDERS = ("orats",)
FEATURES_SOURCE_PROVIDERS = ("orats",)
SERIES_SOURCE_PROVIDERS = ("yfinance",)
RATES_SOURCE_PROVIDERS = ("constant", "fred")

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
OPTIONS_ADAPTER_FACTORIES: dict[str, OptionsAdapterFactory] = {
    "canonical": CanonicalOptionsChainAdapter,
    "orats": OratsOptionsChainAdapter,
    "optionsdx": OptionsDxOptionsChainAdapter,
    "yfinance": YfinanceOptionsChainAdapter,
}


def available_names(registry: dict[str, Callable[..., Any]]) -> str:
    """Return a deterministic comma-joined list of supported registry names."""
    return ", ".join(sorted(registry))
