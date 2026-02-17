"""Black-Scholes pricing engine."""

from __future__ import annotations

from volatility_trading.options.models.black_scholes import bs_greeks, bs_price
from volatility_trading.options.types import MarketState, OptionSpec, PricingResult


class BlackScholesPricer:
    """Exact Black-Scholes pricer backed by analytical formulas."""

    def price(self, spec: OptionSpec, state: MarketState) -> float:
        return bs_price(
            S=state.spot,
            K=spec.strike,
            T=spec.time_to_expiry,
            sigma=state.volatility,
            r=state.rate,
            q=state.dividend_yield,
            option_type=spec.option_type,
        )

    def price_and_greeks(self, spec: OptionSpec, state: MarketState) -> PricingResult:
        out = bs_greeks(
            S=state.spot,
            K=spec.strike,
            T=spec.time_to_expiry,
            sigma=state.volatility,
            r=state.rate,
            q=state.dividend_yield,
            option_type=spec.option_type,
        )
        return PricingResult(**out)
