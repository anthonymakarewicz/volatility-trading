"""Shared risk/margin sizing helpers for arbitrary option structures."""

from __future__ import annotations

from collections.abc import Sequence

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
from .types import EntryIntent, LegSelection


def _effective_leg_side(leg: LegSelection) -> int:
    """Return signed side after applying potential negative leg weights."""
    weight_sign = 1 if leg.spec.weight >= 0 else -1
    return int(leg.side) * weight_sign


def _leg_contract_multiplier(leg: LegSelection, *, lot_size: int) -> float:
    """Return per-leg cash multiplier including lot size and leg ratio."""
    return float(lot_size * abs(int(leg.spec.weight)))


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
        quote_expiry = leg.quote.get("expiry_date")
        if quote_expiry is not None and not pd.isna(quote_expiry):
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
                side=_effective_leg_side(leg),
                contract_multiplier=_leg_contract_multiplier(
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


def size_entry_intent_contracts(
    *,
    intent: EntryIntent,
    lot_size: int,
    spot: float,
    volatility: float,
    equity: float,
    pricer: PriceModel,
    scenario_generator: ScenarioGenerator,
    risk_estimator: RiskEstimator,
    risk_sizer: RiskBudgetSizer | None,
    margin_model: MarginModel | None,
    margin_budget_pct: float | None,
    min_contracts: int,
    max_contracts: int | None,
) -> tuple[int, float | None, str | None, float | None]:
    """Size contracts from risk-budget and margin-budget constraints.

    Returns:
        Tuple ``(contracts, risk_per_contract, risk_scenario, margin_per_contract)``.
    """
    invalid_market = (
        not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(volatility)
        or volatility <= 0
    )
    if invalid_market:
        fallback = risk_sizer.min_contracts if risk_sizer else 1
        return fallback, None, None, None

    option_legs = _build_option_legs(
        entry_date=intent.entry_date,
        default_expiry_date=intent.expiry_date,
        legs=intent.legs,
        lot_size=lot_size,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))

    risk_contracts: int | None = None
    risk_per_contract: float | None = None
    risk_scenario: str | None = None
    if risk_sizer is not None:
        reference_spec = option_legs[0].spec
        scenarios = scenario_generator.generate(spec=reference_spec, state=state)
        if not scenarios:
            risk_contracts = risk_sizer.min_contracts
        else:
            risk_result = risk_estimator.estimate_risk_per_contract(
                legs=option_legs,
                state=state,
                scenarios=scenarios,
                pricer=pricer,
            )
            risk_contracts = risk_sizer.size(
                equity=equity,
                risk_per_contract=risk_result.worst_loss,
            )
            risk_per_contract = risk_result.worst_loss
            risk_scenario = risk_result.worst_scenario.name

    margin_contracts: int | None = None
    margin_per_contract: float | None = None
    if margin_model is not None:
        margin_per_contract = margin_model.initial_margin_requirement(
            legs=option_legs,
            state=state,
            pricer=pricer,
        )
        margin_budget = margin_budget_pct or 1.0
        margin_contracts = contracts_for_risk_budget(
            equity=equity,
            risk_budget_pct=margin_budget,
            risk_per_contract=margin_per_contract,
            min_contracts=0,
            max_contracts=max_contracts,
        )

    if risk_contracts is not None and margin_contracts is not None:
        contracts = min(risk_contracts, margin_contracts)
    elif risk_contracts is not None:
        contracts = risk_contracts
    elif margin_contracts is not None:
        contracts = margin_contracts
    else:
        contracts = max(min_contracts, 1)
        if max_contracts is not None:
            contracts = min(contracts, max_contracts)

    return contracts, risk_per_contract, risk_scenario, margin_per_contract
