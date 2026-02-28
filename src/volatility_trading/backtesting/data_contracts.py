"""Typed dataset bundles consumed by the backtesting runtime."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    """Typed input datasets consumed by options backtesting runtime."""

    options: pd.DataFrame
    features: pd.DataFrame | None = None
    hedge_market: HedgeMarketData | None = None
    fallback_iv_feature_col: str = "iv_atm"
