"""Typed dataset bundles consumed by the backtesting runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from volatility_trading.backtesting.data_adapters import OptionsChainAdapter


@dataclass(frozen=True)
class HedgeMarketData:
    """External hedge instrument market data used by dynamic hedging."""

    mid: pd.Series
    bid: pd.Series | None = None
    ask: pd.Series | None = None
    symbol: str | None = None
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be > 0")


@dataclass(frozen=True)
class OptionsBacktestDataBundle:
    """Typed input datasets consumed by options backtesting runtime.

    ``options_adapter`` is optional. When provided, it normalizes/validates the
    options panel before execution plan compilation. Run-level adapter
    configuration can also be set in ``BacktestRunConfig``.
    """

    options: pd.DataFrame
    features: pd.DataFrame | None = None
    hedge_market: HedgeMarketData | None = None
    fallback_iv_feature_col: str = "iv_atm"
    options_adapter: OptionsChainAdapter | None = None
