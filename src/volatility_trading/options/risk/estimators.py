"""Stress-risk estimators for option positions.

These estimators convert scenario-level shocked repricing into a scalar
`risk_per_contract` used by sizing and margin proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from volatility_trading.options.engines import PriceModel
from volatility_trading.options.models import normalize_option_type
from volatility_trading.options.risk.types import (
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.options.types import (
    MarketState,
    OptionLeg,
    OptionSpec,
    OptionType,
    OptionTypeInput,
)


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


def _leg_shocked_state(
    state: MarketState,
    scenario: StressScenario,
    *,
    option_type: OptionTypeInput,
    min_spot: float,
    min_volatility: float,
) -> MarketState:
    """Apply common shocks plus RR wing shock for one option type."""
    shocked_state = _apply_state_shock(
        state,
        scenario,
        min_spot=min_spot,
        min_volatility=min_volatility,
    )
    rr_shift = 0.5 * scenario.shock.d_risk_reversal
    if normalize_option_type(option_type) == OptionType.PUT:
        rr_shift = -rr_shift
    return MarketState(
        spot=shocked_state.spot,
        volatility=max(shocked_state.volatility + rr_shift, min_volatility),
        rate=shocked_state.rate,
        dividend_yield=shocked_state.dividend_yield,
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
    Parallel vol shocks apply to all legs through `d_volatility`, while
    `d_risk_reversal` applies an additional call-minus-put wing shift at the
    leg level.
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
            pnl = 0.0
            for leg in legs:
                shocked_spec = _apply_time_shock(
                    leg.spec,
                    scenario,
                    min_time_to_expiry=self.min_time_to_expiry,
                )
                shocked_price = pricer.price(
                    shocked_spec,
                    _leg_shocked_state(
                        state,
                        scenario,
                        option_type=leg.spec.option_type,
                        min_spot=self.min_spot,
                        min_volatility=self.min_volatility,
                    ),
                )
                pnl += (
                    leg.side
                    * (shocked_price - leg.entry_price)
                    * leg.contract_multiplier
                )

            points.append(StressPoint(scenario=scenario, pnl=pnl))

        # Most negative PnL determines the risk statistic used for sizing.
        return StressResult.from_points(points)
