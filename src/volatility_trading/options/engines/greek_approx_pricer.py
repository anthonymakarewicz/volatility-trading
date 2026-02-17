"""Sensitivity-based approximation pricer for small shocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from volatility_trading.options.engines.bs_pricer import BlackScholesPricer
from volatility_trading.options.types import (
    MarketShock,
    MarketState,
    OptionSpec,
    PricingResult,
)


@dataclass(frozen=True)
class GreekApproxPricer:
    """Second-order approximation around a reference state.

    This engine is useful when you need fast shocked prices for small moves.
    The canonical exact mark should still use `BlackScholesPricer`.
    """

    base_pricer: BlackScholesPricer = field(default_factory=BlackScholesPricer)

    def price(self, spec: OptionSpec, state: MarketState) -> float:
        """Delegate absolute pricing to exact BS formulas."""
        return self.base_pricer.price(spec, state)

    def price_and_greeks(self, spec: OptionSpec, state: MarketState) -> PricingResult:
        """Delegate exact Greeks to the BS engine."""
        return self.base_pricer.price_and_greeks(spec, state)

    def price_with_shock(
        self,
        spec: OptionSpec,
        reference_state: MarketState,
        shock: MarketShock,
    ) -> float:
        """Approximate shocked price using a local Taylor expansion.

        Terms used:
        - Delta + Gamma in spot
        - Vega in volatility
        - Rho in rates
        - Theta in calendar time
        """
        base = self.base_pricer.price_and_greeks(spec, reference_state)
        d_price = (
            base.delta * shock.d_spot
            + 0.5 * base.gamma * shock.d_spot**2
            + base.vega * shock.d_volatility
            + base.rho * shock.d_rate
            + base.theta * shock.dt_years
        )
        return base.price + d_price
