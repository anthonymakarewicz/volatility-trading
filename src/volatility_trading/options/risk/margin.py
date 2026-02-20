"""Margin model contracts and implementations for options position sizing."""

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
    """Contract for account-style margin requirement estimation."""

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
    """Scenario-based proxy for securities portfolio margin."""

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
