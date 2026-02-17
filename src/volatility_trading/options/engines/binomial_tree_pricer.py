"""Binomial-tree pricing engine for vanilla options."""

from __future__ import annotations

from dataclasses import dataclass

from volatility_trading.options.models.binomial_tree import binomial_tree_price
from volatility_trading.options.types import MarketState, OptionSpec


@dataclass(frozen=True)
class BinomialTreePricer:
    """CRR tree pricer supporting American and European exercise.

    This engine intentionally implements only `price(...)`. Greek estimation can
    be added later through finite-difference wrappers if needed.
    """

    steps: int = 200
    american: bool = True

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")

    def price(self, spec: OptionSpec, state: MarketState) -> float:
        return binomial_tree_price(
            S=state.spot,
            K=spec.strike,
            T=spec.time_to_expiry,
            sigma=state.volatility,
            r=state.rate,
            q=state.dividend_yield,
            option_type=spec.option_type,
            steps=self.steps,
            american=self.american,
        )
