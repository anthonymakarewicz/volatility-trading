"""Risk estimators for option positions under stress scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from volatility_trading.options.engines import PriceModel
from volatility_trading.options.risk.types import (
    OptionLeg,
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.options.types import MarketState, OptionSpec


def _apply_state_shock(
    state: MarketState,
    scenario: StressScenario,
    *,
    min_spot: float,
    min_volatility: float,
) -> MarketState:
    """Return a shocked market state with simple numeric floors."""
    shock = scenario.shock
    return MarketState(
        spot=max(state.spot + shock.d_spot, min_spot),
        volatility=max(state.volatility + shock.d_volatility, min_volatility),
        rate=state.rate + shock.d_rate,
        dividend_yield=state.dividend_yield,
    )


def _apply_time_shock(
    spec: OptionSpec, scenario: StressScenario, *, min_time_to_expiry: float
) -> OptionSpec:
    """Return a shocked option spec after rolling time forward."""
    return OptionSpec(
        strike=spec.strike,
        time_to_expiry=max(
            spec.time_to_expiry - scenario.shock.dt_years, min_time_to_expiry
        ),
        option_type=spec.option_type,
    )


@runtime_checkable
class RiskEstimator(Protocol):
    """Contract for stress-based risk estimators used in position sizing."""

    def estimate_risk_per_contract(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        scenarios: Sequence[StressScenario],
        pricer: PriceModel,
    ) -> StressResult:
        """Return risk summary based on stressed PnL across scenarios."""
        ...


@dataclass(frozen=True)
class StressLossRiskEstimator:
    """Estimate risk per contract as worst-case loss across stress scenarios."""

    min_spot: float = 1e-8
    min_volatility: float = 1e-8
    min_time_to_expiry: float = 1e-8

    def estimate_risk_per_contract(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        scenarios: Sequence[StressScenario],
        pricer: PriceModel,
    ) -> StressResult:
        if not legs:
            raise ValueError("legs must not be empty")
        if not scenarios:
            raise ValueError("scenarios must not be empty")

        points: list[StressPoint] = []

        for scenario in scenarios:
            shocked_state = _apply_state_shock(
                state,
                scenario,
                min_spot=self.min_spot,
                min_volatility=self.min_volatility,
            )

            pnl = 0.0
            for leg in legs:
                shocked_spec = _apply_time_shock(
                    leg.spec,
                    scenario,
                    min_time_to_expiry=self.min_time_to_expiry,
                )
                shocked_price = pricer.price(shocked_spec, shocked_state)
                pnl += (
                    leg.side
                    * (shocked_price - leg.entry_price)
                    * leg.contract_multiplier
                )

            points.append(StressPoint(scenario=scenario, pnl=pnl))

        worst_point = min(points, key=lambda point: point.pnl)
        worst_loss = max(-worst_point.pnl, 0.0)
        return StressResult(
            worst_loss=worst_loss,
            worst_scenario=worst_point.scenario,
            points=tuple(points),
        )
