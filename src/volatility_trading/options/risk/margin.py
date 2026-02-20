"""Position-level margin requirement models for option risk sizing.

This module estimates *initial* margin per position unit and is intended for
entry sizing / capacity checks, not full account lifecycle simulation.

Scope:
- estimate initial margin for a given set of option legs at a market state
- provide deterministic Reg-T style approximation
- provide scenario-based Portfolio Margin proxy

Out of scope:
- maintenance margin progression through time
- margin-call day counting and liquidation workflow
- financing carry and cash/borrow account mechanics

Use :class:`PortfolioMarginProxyModel` when your account behaves like Portfolio
Margin and you want stress-based sizing that better reflects concentration and
scenario risk than formula-style Reg-T approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from volatility_trading.options.engines import PriceModel
from volatility_trading.options.risk.estimators import (
    RiskEstimator,
    StressLossRiskEstimator,
)
from volatility_trading.options.risk.scenarios import (
    FixedGridScenarioGenerator,
    ScenarioGenerator,
)
from volatility_trading.options.risk.types import OptionLeg, PositionSide
from volatility_trading.options.types import MarketState, OptionType


@runtime_checkable
class MarginModel(Protocol):
    """Contract for position-level initial margin estimation.

    Implementations return the margin requirement for one position unit
    (typically one strategy unit that may contain multiple legs).
    """

    def initial_margin_requirement(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        """Return initial margin requirement for one position unit."""
        ...


@dataclass(frozen=True)
class RegTMarginModel:
    """Approximate Reg-T/strategy-based margin for listed options.

    Notes:
        - This is a pragmatic backtesting approximation, not a broker-legal
          implementation.
        - House overlays are modeled via `house_multiplier` and `house_floor`.
        - For two-leg short call+put structures, this model applies a
          "greater side + other option value" simplification.

    Attributes:
        broad_index: If True, uses a lower base short-option percent intended
            for broad index products.
        minimum_underlying_pct: Floor for short-call requirement as percent of
            spot notional.
        minimum_put_pct: Floor for short-put requirement as percent of strike
            notional.
        house_multiplier: Broker house overlay multiplier (>= 1).
        house_floor: Absolute broker house floor.
    """

    broad_index: bool = False
    minimum_underlying_pct: float = 0.10
    minimum_put_pct: float = 0.10
    house_multiplier: float = 1.0
    house_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_underlying_pct < 0:
            raise ValueError("minimum_underlying_pct must be >= 0")
        if self.minimum_put_pct < 0:
            raise ValueError("minimum_put_pct must be >= 0")
        if self.house_multiplier < 1.0:
            raise ValueError("house_multiplier must be >= 1.0")
        if self.house_floor < 0:
            raise ValueError("house_floor must be >= 0")

    @property
    def base_pct(self) -> float:
        """Base short-option percentage by product family."""
        return 0.15 if self.broad_index else 0.20

    def initial_margin_requirement(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        """Compute Reg-T style initial margin for one position unit.

        The implementation handles:
        - generic sum of per-leg requirements
        - short call+put pair shortcut via "greater side + other option value"
          approximation.
        """
        if not legs:
            raise ValueError("legs must not be empty")

        base_requirement = self._base_requirement(legs=legs, state=state, pricer=pricer)
        return max(base_requirement * self.house_multiplier, self.house_floor)

    def _base_requirement(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        if len(legs) == 2 and all(leg.side == PositionSide.SHORT for leg in legs):
            if self._is_short_call_put_pair(legs):
                return self._short_pair_requirement(
                    legs=legs, state=state, pricer=pricer
                )

        return sum(
            self._single_leg_requirement(leg=leg, state=state, pricer=pricer)
            for leg in legs
        )

    @staticmethod
    def _is_short_call_put_pair(legs: Sequence[OptionLeg]) -> bool:
        option_types = {leg.spec.option_type for leg in legs}
        return option_types == {OptionType.CALL, OptionType.PUT}

    def _short_pair_requirement(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        call_leg = next(leg for leg in legs if leg.spec.option_type == OptionType.CALL)
        put_leg = next(leg for leg in legs if leg.spec.option_type == OptionType.PUT)

        call_req = self._short_call_requirement(
            leg=call_leg, state=state, pricer=pricer
        )
        put_req = self._short_put_requirement(leg=put_leg, state=state, pricer=pricer)
        call_value = self._option_value(leg=call_leg, state=state, pricer=pricer)
        put_value = self._option_value(leg=put_leg, state=state, pricer=pricer)

        return max(call_req + put_value, put_req + call_value)

    def _single_leg_requirement(
        self,
        *,
        leg: OptionLeg,
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        if leg.side == PositionSide.LONG:
            return self._option_value(leg=leg, state=state, pricer=pricer)
        if leg.spec.option_type == OptionType.CALL:
            return self._short_call_requirement(leg=leg, state=state, pricer=pricer)
        return self._short_put_requirement(leg=leg, state=state, pricer=pricer)

    def _short_call_requirement(
        self,
        *,
        leg: OptionLeg,
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        option_value = self._option_value(leg=leg, state=state, pricer=pricer)
        otm_amount = max(leg.spec.strike - state.spot, 0.0) * leg.contract_multiplier
        base_component = self.base_pct * state.spot * leg.contract_multiplier
        minimum_component = (
            self.minimum_underlying_pct * state.spot * leg.contract_multiplier
        )
        return option_value + max(base_component - otm_amount, minimum_component)

    def _short_put_requirement(
        self,
        *,
        leg: OptionLeg,
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        option_value = self._option_value(leg=leg, state=state, pricer=pricer)
        otm_amount = max(state.spot - leg.spec.strike, 0.0) * leg.contract_multiplier
        base_component = self.base_pct * state.spot * leg.contract_multiplier
        minimum_component = (
            self.minimum_put_pct * leg.spec.strike * leg.contract_multiplier
        )
        return option_value + max(base_component - otm_amount, minimum_component)

    @staticmethod
    def _option_value(
        *,
        leg: OptionLeg,
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        return pricer.price(leg.spec, state) * leg.contract_multiplier


@dataclass(frozen=True)
class PortfolioMarginProxyModel:
    """Scenario-based proxy for securities Portfolio Margin.

    This is generally the better choice when you expect PM-like behavior in live
    trading and want more realistic stress-based sizing than fixed Reg-T
    formulas.

    Process:
    1) Generate stress scenarios.
    2) Compute worst stressed loss for the provided legs.
    3) Convert worst loss to margin via multiplier/floor overlays.

    Attributes:
        scenario_generator: Produces stress scenarios for revaluation.
        risk_estimator: Computes worst-loss across provided scenarios.
        stress_multiplier: Multiplier on worst loss (>= 1 is conservative).
        minimum_margin: Absolute minimum requirement before house overlays.
        house_multiplier: Broker house overlay multiplier (>= 1).
        house_floor: Absolute broker house floor.
    """

    scenario_generator: ScenarioGenerator = FixedGridScenarioGenerator()
    risk_estimator: RiskEstimator = StressLossRiskEstimator()
    stress_multiplier: float = 1.0
    minimum_margin: float = 0.0
    house_multiplier: float = 1.0
    house_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.stress_multiplier <= 0:
            raise ValueError("stress_multiplier must be > 0")
        if self.minimum_margin < 0:
            raise ValueError("minimum_margin must be >= 0")
        if self.house_multiplier < 1.0:
            raise ValueError("house_multiplier must be >= 1.0")
        if self.house_floor < 0:
            raise ValueError("house_floor must be >= 0")

    def initial_margin_requirement(
        self,
        *,
        legs: Sequence[OptionLeg],
        state: MarketState,
        pricer: PriceModel,
    ) -> float:
        """Compute scenario-based initial margin requirement.

        Returns:
            Margin requirement for one position unit derived from stressed PnL.
        """
        if not legs:
            raise ValueError("legs must not be empty")

        reference_spec = legs[0].spec
        scenarios = self.scenario_generator.generate(spec=reference_spec, state=state)
        if not scenarios:
            raise ValueError("scenario generator returned no scenarios")

        stress = self.risk_estimator.estimate_risk_per_contract(
            legs=legs,
            state=state,
            scenarios=scenarios,
            pricer=pricer,
        )

        base = max(stress.worst_loss * self.stress_multiplier, self.minimum_margin)
        return max(base * self.house_multiplier, self.house_floor)
