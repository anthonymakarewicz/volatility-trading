"""Interface for option-pricing engines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from volatility_trading.options.types import MarketState, OptionSpec, PricingResult


@runtime_checkable
class PriceModel(Protocol):
    """Minimum pricing capability required by strategies and risk modules."""

    def price(self, spec: OptionSpec, state: MarketState) -> float:
        """Return option value for one contract."""


@runtime_checkable
class GreeksModel(Protocol):
    """Optional extension for engines that provide stable sensitivities."""

    def price_and_greeks(self, spec: OptionSpec, state: MarketState) -> PricingResult:
        """Return option value and sensitivities for one contract."""
