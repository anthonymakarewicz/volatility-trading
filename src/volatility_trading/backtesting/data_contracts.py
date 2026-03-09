"""Typed dataset bundles consumed by the backtesting runtime."""

from __future__ import annotations

import math
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
        if not math.isfinite(self.contract_multiplier) or self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be finite and > 0")


@dataclass(frozen=True)
class OptionsMarketData:
    """Options-chain market dataset plus chain-level metadata.

    Attributes:
        chain: Option chain panel for one backtest run.
        symbol: Optional underlying ticker label for reporting/validation hooks.
        default_contract_multiplier: Fallback option contract multiplier when
            quote-level multipliers are not modeled in the chain schema.
        options_adapter: Optional adapter scoped specifically to this chain.
    """

    chain: pd.DataFrame
    symbol: str | None = None
    default_contract_multiplier: float = 100.0
    options_adapter: OptionsChainAdapter | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.default_contract_multiplier)
            or self.default_contract_multiplier <= 0
        ):
            raise ValueError("default_contract_multiplier must be finite and > 0")


@dataclass(frozen=True)
class OptionsBacktestDataBundle:
    """Typed input datasets consumed by options backtesting runtime.

    ``options_market`` carries the options chain plus optional chain-level
    metadata (symbol, default multiplier, scoped adapter).
    """

    options_market: OptionsMarketData
    features: pd.DataFrame | None = None
    hedge_market: HedgeMarketData | None = None
    fallback_iv_feature_col: str = "iv_atm"  # TODO: Remove it

    @property
    def options_frame(self) -> pd.DataFrame:
        """Return options panel normalized from the configured bundle source."""
        return self.options_market.chain

    @property
    def option_symbol(self) -> str | None:
        """Return optional options underlying symbol metadata."""
        return self.options_market.symbol

    @property
    def option_contract_multiplier(self) -> float:
        """Return bundle-level default option contract multiplier metadata."""
        return float(self.options_market.default_contract_multiplier)
