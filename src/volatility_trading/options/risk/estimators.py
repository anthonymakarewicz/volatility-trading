"""Stress-risk estimators for option positions.

These estimators convert scenario-level shocked repricing into a scalar
`risk_per_contract` used by sizing and margin proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from volatility_trading.options.engines import PriceModel
from volatility_trading.options.risk.types import (
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.options.types import MarketState, OptionLeg, OptionSpec


def _apply_state_shock(
    state: MarketState,
    scenario: StressScenario,
    *,
    min_spot: float,
    min_volatility: float,
) -> MarketState:
    """Apply scenario shocks to market state while enforcing numeric floors."""
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
    """Roll time forward by scenario `dt_years` with a minimum expiry floor."""
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
        """Return stressed-PnL summary and worst-loss statistic."""
        ...


@dataclass(frozen=True)
class StressLossRiskEstimator:
    """Estimate risk as worst-case loss across provided stress scenarios.

    For each scenario, legs are repriced on shocked state/spec and aggregated as:
    `pnl = sum(side * (shocked_price - entry_price) * contract_multiplier)`.
    Final risk metric is `worst_loss = max(-min(pnl), 0)`.
    """

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
        """Compute scenario PnL points and return the worst-loss summary."""
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

        # Most negative PnL determines the risk statistic used for sizing.
        worst_point = min(points, key=lambda point: point.pnl)
        worst_loss = max(-worst_point.pnl, 0.0)
        return StressResult(
            worst_loss=worst_loss,
            worst_scenario=worst_point.scenario,
            points=tuple(points),
        )
