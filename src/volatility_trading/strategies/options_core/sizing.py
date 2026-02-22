"""Shared risk/margin sizing helpers for arbitrary option structures.

The primary APIs in this module are structure-agnostic and operate on
`LegSelection` / `EntryIntent` payloads. Legacy short-straddle wrappers are
kept to preserve existing behavior while strategies migrate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from volatility_trading.options import (
    MarginModel,
    MarketState,
    OptionType,
    PriceModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    contracts_for_risk_budget,
)

from .adapters import quote_to_option_leg
from .types import EntryIntent, LegSelection, LegSpec


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
    expiry_date: pd.Timestamp,
    legs: Sequence[LegSelection],
    lot_size: int,
):
    if not legs:
        raise ValueError("legs must not be empty")

    return tuple(
        quote_to_option_leg(
            quote=leg.quote,
            entry_date=entry_date,
            expiry_date=expiry_date,
            entry_price=leg.entry_price,
            side=_effective_leg_side(leg),
            contract_multiplier=_leg_contract_multiplier(leg, lot_size=lot_size),
        )
        for leg in legs
    )


def estimate_structure_margin_per_contract(
    *,
    as_of_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    legs: Sequence[LegSelection],
    lot_size: int,
    spot: float,
    volatility: float,
    margin_model: MarginModel | None,
    pricer: PriceModel,
) -> float | None:
    """Estimate initial margin for one structure unit as of `as_of_date`."""
    if margin_model is None:
        return None
    if not np.isfinite(spot) or spot <= 0:
        return None
    if not np.isfinite(volatility) or volatility <= 0:
        return None

    option_legs = _build_option_legs(
        entry_date=as_of_date,
        expiry_date=expiry_date,
        legs=legs,
        lot_size=lot_size,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))
    return margin_model.initial_margin_requirement(
        legs=option_legs,
        state=state,
        pricer=pricer,
    )


def size_structure_contracts(
    *,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    legs: Sequence[LegSelection],
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
    """Size contracts from risk and margin budgets for arbitrary structures."""
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
        entry_date=entry_date,
        expiry_date=expiry_date,
        legs=legs,
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
    """Estimate margin from one `EntryIntent` payload."""
    return estimate_structure_margin_per_contract(
        as_of_date=as_of_date or intent.entry_date,
        expiry_date=intent.expiry_date,
        legs=intent.legs,
        lot_size=lot_size,
        spot=spot,
        volatility=volatility,
        margin_model=margin_model,
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
    """Size contracts directly from an `EntryIntent` payload."""
    return size_structure_contracts(
        entry_date=intent.entry_date,
        expiry_date=intent.expiry_date,
        legs=intent.legs,
        lot_size=lot_size,
        spot=spot,
        volatility=volatility,
        equity=equity,
        pricer=pricer,
        scenario_generator=scenario_generator,
        risk_estimator=risk_estimator,
        risk_sizer=risk_sizer,
        margin_model=margin_model,
        margin_budget_pct=margin_budget_pct,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
    )


def _short_straddle_intent(
    *,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    put_quote: pd.Series,
    call_quote: pd.Series,
    put_entry: float,
    call_entry: float,
) -> EntryIntent:
    chosen_dte = int(
        put_quote.get("dte")
        if pd.notna(put_quote.get("dte"))
        else call_quote.get("dte")
        if pd.notna(call_quote.get("dte"))
        else max((pd.Timestamp(expiry_date) - pd.Timestamp(entry_date)).days, 1)
    )
    put_leg = LegSelection(
        spec=LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
        quote=put_quote,
        side=-1,
        entry_price=float(put_entry),
    )
    call_leg = LegSelection(
        spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
        quote=call_quote,
        side=-1,
        entry_price=float(call_entry),
    )
    return EntryIntent(
        entry_date=entry_date,
        expiry_date=expiry_date,
        chosen_dte=chosen_dte,
        legs=(put_leg, call_leg),
    )


def estimate_short_straddle_margin_per_contract(
    *,
    as_of_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    put_quote: pd.Series,
    call_quote: pd.Series,
    put_entry: float,
    call_entry: float,
    lot_size: int,
    spot: float,
    volatility: float,
    margin_model: MarginModel | None,
    pricer: PriceModel,
) -> float | None:
    """Backward-compatible wrapper for short-straddle margin estimation."""
    intent = _short_straddle_intent(
        entry_date=as_of_date,
        expiry_date=expiry_date,
        put_quote=put_quote,
        call_quote=call_quote,
        put_entry=put_entry,
        call_entry=call_entry,
    )
    return estimate_entry_intent_margin_per_contract(
        intent=intent,
        as_of_date=as_of_date,
        lot_size=lot_size,
        spot=spot,
        volatility=volatility,
        margin_model=margin_model,
        pricer=pricer,
    )


def size_short_straddle_contracts(
    *,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    put_quote: pd.Series,
    call_quote: pd.Series,
    put_entry: float,
    call_entry: float,
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
    """Backward-compatible wrapper for short-straddle sizing."""
    intent = _short_straddle_intent(
        entry_date=entry_date,
        expiry_date=expiry_date,
        put_quote=put_quote,
        call_quote=call_quote,
        put_entry=put_entry,
        call_entry=call_entry,
    )
    return size_entry_intent_contracts(
        intent=intent,
        lot_size=lot_size,
        spot=spot,
        volatility=volatility,
        equity=equity,
        pricer=pricer,
        scenario_generator=scenario_generator,
        risk_estimator=risk_estimator,
        risk_sizer=risk_sizer,
        margin_model=margin_model,
        margin_budget_pct=margin_budget_pct,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
    )
