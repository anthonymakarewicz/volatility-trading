"""Shared risk/margin sizing helpers for arbitrary option structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volatility_trading.options import (
    MarginModel,
    MarketState,
    PriceModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    contracts_for_risk_budget,
)

from .adapters import quote_to_option_leg
from .economics import effective_leg_side, leg_contract_multiplier
from .types import EntryIntent, LegSelection


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Inputs required to size one entry intent from risk/margin constraints."""

    intent: EntryIntent
    lot_size: int
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

    def __post_init__(self) -> None:
        if self.min_contracts < 0:
            raise ValueError("min_contracts must be >= 0")
        if self.max_contracts is not None and self.max_contracts <= 0:
            raise ValueError("max_contracts must be > 0 when provided")
        if self.max_contracts is not None and self.max_contracts < self.min_contracts:
            raise ValueError("max_contracts must be >= min_contracts")
        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SizingDecision:
    """Resolved contract size and diagnostics for one entry intent."""

    contracts: int
    risk_per_contract: float | None
    risk_scenario: str | None
    margin_per_contract: float | None


def _build_option_legs(
    *,
    entry_date: pd.Timestamp,
    default_expiry_date: pd.Timestamp | None,
    legs: Sequence[LegSelection],
    lot_size: int,
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
                    lot_size=lot_size,
                ),
            )
        )
    return tuple(built_legs)


def estimate_entry_intent_margin_per_contract(
    *,
    intent: EntryIntent,
    as_of_date: pd.Timestamp | None,
    lot_size: int,
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
        lot_size=lot_size,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))
    return margin_model.initial_margin_requirement(
        legs=option_legs,
        state=state,
        pricer=pricer,
    )


def size_entry_intent(request: SizingRequest) -> SizingDecision:
    """Size contracts from risk-budget and margin-budget constraints."""
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
        )

    option_legs = _build_option_legs(
        entry_date=request.intent.entry_date,
        default_expiry_date=request.intent.expiry_date,
        legs=request.intent.legs,
        lot_size=request.lot_size,
    )
    state = MarketState(spot=float(request.spot), volatility=float(request.volatility))

    risk_contracts: int | None = None
    risk_per_contract: float | None = None
    risk_scenario: str | None = None
    if request.risk_sizer is not None:
        reference_spec = option_legs[0].spec
        scenarios = request.scenario_generator.generate(
            spec=reference_spec, state=state
        )
        if not scenarios:
            risk_contracts = request.risk_sizer.min_contracts
        else:
            risk_result = request.risk_estimator.estimate_risk_per_contract(
                legs=option_legs,
                state=state,
                scenarios=scenarios,
                pricer=request.pricer,
            )
            risk_contracts = request.risk_sizer.size(
                equity=request.equity,
                risk_per_contract=risk_result.worst_loss,
            )
            risk_per_contract = risk_result.worst_loss
            risk_scenario = risk_result.worst_scenario.name

    margin_contracts: int | None = None
    margin_per_contract: float | None = None
    if request.margin_model is not None:
        margin_per_contract = request.margin_model.initial_margin_requirement(
            legs=option_legs,
            state=state,
            pricer=request.pricer,
        )
        margin_budget = request.margin_budget_pct or 1.0
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

    return SizingDecision(
        contracts=contracts,
        risk_per_contract=risk_per_contract,
        risk_scenario=risk_scenario,
        margin_per_contract=margin_per_contract,
    )
