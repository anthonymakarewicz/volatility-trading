"""Shared risk/margin sizing helpers for arbitrary option structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from volatility_trading.options import (
    MarginModel,
    MarketState,
    PriceModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressPoint,
    StressResult,
    contracts_for_risk_budget,
)

from .adapters import quote_to_option_leg
from .contracts.structures import EntryIntent, LegSelection
from .economics import effective_leg_side, leg_contract_multiplier


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Inputs required to size one entry intent from risk/margin constraints."""

    intent: EntryIntent
    option_contract_multiplier: float
    spot: float
    volatility: float
    equity: float
    pricer: PriceModel
    scenario_generator: ScenarioGenerator
    risk_estimator: RiskEstimator
    risk_sizer: RiskBudgetSizer | None
    margin_model: MarginModel | None
    margin_budget_pct: float | None
    min_contracts: int
    max_contracts: int | None
    entry_risk_basis: Literal["unhedged", "entry_hedged"] = "unhedged"
    entry_hedge_target_net_delta: float = 0.0
    hedge_contract_multiplier: float | None = None

    def __post_init__(self) -> None:
        if self.min_contracts < 0:
            raise ValueError("min_contracts must be >= 0")
        if self.max_contracts is not None and self.max_contracts <= 0:
            raise ValueError("max_contracts must be > 0 when provided")
        if self.max_contracts is not None and self.max_contracts < self.min_contracts:
            raise ValueError("max_contracts must be >= min_contracts")
        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")
        if self.entry_risk_basis not in {"unhedged", "entry_hedged"}:
            raise ValueError("entry_risk_basis must be 'unhedged' or 'entry_hedged'")
        if (
            self.entry_risk_basis == "entry_hedged"
            and self.risk_sizer is not None
            and (
                self.hedge_contract_multiplier is None
                or not np.isfinite(self.hedge_contract_multiplier)
                or self.hedge_contract_multiplier <= 0
            )
        ):
            raise ValueError(
                "entry_hedged sizing requires hedge_contract_multiplier > 0"
            )


@dataclass(frozen=True, slots=True)
class SizingDecision:
    """Resolved contract size and diagnostics for one entry intent.

    Attributes:
        contracts: Final contract count after all sizing constraints and policy
            overrides are applied.
        risk_per_contract: Worst-loss estimate per one-lot option structure
            from the configured risk estimator, if available.
        risk_scenario: Name of the stress scenario producing
            `risk_per_contract`, if available.
        margin_per_contract: Initial margin estimate per one-lot option
            structure, if available.
        risk_budget_contracts: Raw contract count allowed by the risk budget
            alone before any `min_contracts` floor is applied.
        margin_budget_contracts: Raw contract count allowed by the margin
            budget alone before the final risk-vs-margin minimum is taken.
        sizing_binding_constraint: Label describing which constraint determined
            the final `contracts` value (for example `risk_budget`,
            `margin_budget`, or `min_contracts`).
        min_contracts_override_applied: Whether a nonzero `min_contracts`
            floor increased the risk-sized contract count above the raw
            `risk_budget_contracts` result.
    """

    contracts: int
    risk_per_contract: float | None
    risk_scenario: str | None
    margin_per_contract: float | None
    risk_budget_contracts: int | None = None
    margin_budget_contracts: int | None = None
    sizing_binding_constraint: str | None = None
    min_contracts_override_applied: bool = False


def _build_option_legs(
    *,
    entry_date: pd.Timestamp,
    default_expiry_date: pd.Timestamp | None,
    legs: Sequence[LegSelection],
    option_contract_multiplier: float,
):
    """Convert selected strategy legs into option-risk engine ``OptionLeg`` rows.

    Raises:
        ValueError: If no legs are provided or expiry cannot be resolved for a leg.
    """
    if not legs:
        raise ValueError("legs must not be empty")

    built_legs = []
    for leg in legs:
        quote_expiry = leg.quote.expiry_date
        if quote_expiry is not None:
            expiry_date = pd.Timestamp(quote_expiry)
        elif default_expiry_date is not None:
            expiry_date = pd.Timestamp(default_expiry_date)
        else:
            raise ValueError(
                "Missing expiry_date on leg quote and no default_expiry_date provided."
            )

        built_legs.append(
            quote_to_option_leg(
                quote=leg.quote,
                entry_date=entry_date,
                expiry_date=expiry_date,
                entry_price=leg.entry_price,
                side=effective_leg_side(leg),
                contract_multiplier=leg_contract_multiplier(
                    leg,
                    option_contract_multiplier=option_contract_multiplier,
                ),
            )
        )
    return tuple(built_legs)


def _entry_option_delta_per_contract(
    *,
    legs: Sequence[LegSelection],
    option_contract_multiplier: float,
) -> float:
    """Return structure delta for one strategy contract at entry."""
    return float(
        sum(
            effective_leg_side(leg)
            * float(leg.quote.delta)
            * leg_contract_multiplier(
                leg,
                option_contract_multiplier=option_contract_multiplier,
            )
            for leg in legs
        )
    )


def _entry_hedged_stress_result(
    *,
    option_stress: StressResult,
    option_delta_per_contract: float,
    target_net_delta: float,
    hedge_contract_multiplier: float,
) -> StressResult:
    """Return stressed risk after adding the inception hedge package PnL."""
    hedge_qty = (float(target_net_delta) - float(option_delta_per_contract)) / float(
        hedge_contract_multiplier
    )
    hedged_points = tuple(
        StressPoint(
            scenario=point.scenario,
            pnl=(
                float(point.pnl)
                + hedge_qty
                * float(hedge_contract_multiplier)
                * float(point.scenario.shock.d_spot)
            ),
        )
        for point in option_stress.points
    )
    worst_point = min(hedged_points, key=lambda point: point.pnl)
    return StressResult(
        worst_loss=max(-worst_point.pnl, 0.0),
        worst_scenario=worst_point.scenario,
        points=hedged_points,
    )


def estimate_entry_intent_margin_per_contract(
    *,
    intent: EntryIntent,
    as_of_date: pd.Timestamp | None,
    option_contract_multiplier: float,
    spot: float,
    volatility: float,
    margin_model: MarginModel | None,
    pricer: PriceModel,
) -> float | None:
    """Estimate initial margin for one `EntryIntent` contract unit."""
    if margin_model is None:
        return None
    if not np.isfinite(spot) or spot <= 0:
        return None
    if not np.isfinite(volatility) or volatility <= 0:
        return None

    option_legs = _build_option_legs(
        entry_date=as_of_date or intent.entry_date,
        default_expiry_date=intent.expiry_date,
        legs=intent.legs,
        option_contract_multiplier=option_contract_multiplier,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))
    return margin_model.initial_margin_requirement(
        legs=option_legs,
        state=state,
        pricer=pricer,
    )


def size_entry_intent(request: SizingRequest) -> SizingDecision:
    """Size contracts from risk-budget and margin-budget constraints.

    The returned `contracts` value is the final execution size after combining
    risk and margin limits with any configured `min_contracts` / `max_contracts`
    policy clamps. Use `risk_budget_contracts` and `margin_budget_contracts`
    when you need the unclamped raw budget-limited sizes for diagnostics.

    When `entry_risk_basis="entry_hedged"`, `risk_per_contract` is computed on
    the option package plus the implied inception hedge needed to reach
    `entry_hedge_target_net_delta`.

    Invalid spot/volatility inputs skip stress and margin estimation entirely
    and fall back to the configured minimum-lot policy.
    """
    invalid_market = (
        not np.isfinite(request.spot)
        or request.spot <= 0
        or not np.isfinite(request.volatility)
        or request.volatility <= 0
    )
    if invalid_market:
        fallback = request.risk_sizer.min_contracts if request.risk_sizer else 1
        return SizingDecision(
            contracts=fallback,
            risk_per_contract=None,
            risk_scenario=None,
            margin_per_contract=None,
            sizing_binding_constraint="invalid_market",
        )

    option_legs = _build_option_legs(
        entry_date=request.intent.entry_date,
        default_expiry_date=request.intent.expiry_date,
        legs=request.intent.legs,
        option_contract_multiplier=request.option_contract_multiplier,
    )
    state = MarketState(spot=float(request.spot), volatility=float(request.volatility))

    risk_contracts: int | None = None
    risk_per_contract: float | None = None
    risk_scenario: str | None = None
    risk_budget_contracts: int | None = None
    min_contracts_override_applied = False
    if request.risk_sizer is not None:
        reference_spec = option_legs[0].spec
        scenarios = request.scenario_generator.generate(
            spec=reference_spec, state=state
        )
        if not scenarios:
            risk_contracts = request.risk_sizer.min_contracts
        else:
            option_risk_result = request.risk_estimator.estimate_risk_per_contract(
                legs=option_legs,
                state=state,
                scenarios=scenarios,
                pricer=request.pricer,
            )
            risk_result = option_risk_result
            if request.entry_risk_basis == "entry_hedged":
                risk_result = _entry_hedged_stress_result(
                    option_stress=option_risk_result,
                    option_delta_per_contract=_entry_option_delta_per_contract(
                        legs=request.intent.legs,
                        option_contract_multiplier=request.option_contract_multiplier,
                    ),
                    target_net_delta=request.entry_hedge_target_net_delta,
                    hedge_contract_multiplier=float(request.hedge_contract_multiplier),
                )
            risk_budget_contracts = contracts_for_risk_budget(
                equity=request.equity,
                risk_budget_pct=request.risk_sizer.risk_budget_pct,
                risk_per_contract=risk_result.worst_loss,
                min_contracts=0,
                max_contracts=None,
            )
            risk_contracts = request.risk_sizer.size(
                equity=request.equity,
                risk_per_contract=risk_result.worst_loss,
            )
            risk_per_contract = risk_result.worst_loss
            risk_scenario = risk_result.worst_scenario.name
            min_contracts_override_applied = (
                risk_budget_contracts is not None
                and risk_contracts > risk_budget_contracts
            )

    margin_contracts: int | None = None
    margin_per_contract: float | None = None
    margin_budget_contracts: int | None = None
    if request.margin_model is not None:
        margin_per_contract = request.margin_model.initial_margin_requirement(
            legs=option_legs,
            state=state,
            pricer=request.pricer,
        )
        margin_budget = request.margin_budget_pct or 1.0
        margin_budget_contracts = contracts_for_risk_budget(
            equity=request.equity,
            risk_budget_pct=margin_budget,
            risk_per_contract=margin_per_contract,
            min_contracts=0,
            max_contracts=None,
        )
        margin_contracts = contracts_for_risk_budget(
            equity=request.equity,
            risk_budget_pct=margin_budget,
            risk_per_contract=margin_per_contract,
            min_contracts=0,
            max_contracts=request.max_contracts,
        )

    if risk_contracts is not None and margin_contracts is not None:
        contracts = min(risk_contracts, margin_contracts)
    elif risk_contracts is not None:
        contracts = risk_contracts
    elif margin_contracts is not None:
        contracts = margin_contracts
    else:
        contracts = max(request.min_contracts, 1)
        if request.max_contracts is not None:
            contracts = min(contracts, request.max_contracts)

    sizing_binding_constraint = _resolve_sizing_binding_constraint(
        contracts=contracts,
        risk_contracts=risk_contracts,
        margin_contracts=margin_contracts,
        risk_budget_contracts=risk_budget_contracts,
        margin_budget_contracts=margin_budget_contracts,
        min_contracts_override_applied=min_contracts_override_applied,
        max_contracts=request.max_contracts,
    )

    return SizingDecision(
        contracts=contracts,
        risk_per_contract=risk_per_contract,
        risk_scenario=risk_scenario,
        margin_per_contract=margin_per_contract,
        risk_budget_contracts=risk_budget_contracts,
        margin_budget_contracts=margin_budget_contracts,
        sizing_binding_constraint=sizing_binding_constraint,
        min_contracts_override_applied=min_contracts_override_applied,
    )


def _resolve_sizing_binding_constraint(
    *,
    contracts: int,
    risk_contracts: int | None,
    margin_contracts: int | None,
    risk_budget_contracts: int | None,
    margin_budget_contracts: int | None,
    min_contracts_override_applied: bool,
    max_contracts: int | None,
) -> str:
    """Return the dominant sizing constraint label for one sizing decision.

    Constraint precedence is resolved in this order:
    `max_contracts`, then `min_contracts`, then the tighter of the risk and
    margin budget constraints.
    """
    if (
        max_contracts is not None
        and contracts == max_contracts
        and any(
            raw_contracts is not None and raw_contracts > max_contracts
            for raw_contracts in (risk_budget_contracts, margin_budget_contracts)
        )
    ):
        return "max_contracts"

    if min_contracts_override_applied and (
        margin_contracts is None or contracts <= margin_contracts
    ):
        return "min_contracts"

    if risk_contracts is not None and margin_contracts is not None:
        if risk_contracts < margin_contracts:
            return "risk_budget"
        if margin_contracts < risk_contracts:
            return "margin_budget"
        return "risk_and_margin_budget"

    if risk_contracts is not None:
        return "risk_budget"
    if margin_contracts is not None:
        return "margin_budget"
    return "unconstrained"
