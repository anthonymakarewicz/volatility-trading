"""Shared risk/margin contract-sizing helpers for options structures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_trading.options import (
    MarginModel,
    MarketState,
    OptionLeg,
    OptionSpec,
    OptionType,
    PositionSide,
    PriceModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    contracts_for_risk_budget,
)

from .adapters import time_to_expiry_years


def _build_short_straddle_legs(
    *,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    put_quote: pd.Series,
    call_quote: pd.Series,
    put_entry: float,
    call_entry: float,
    lot_size: int,
) -> tuple[OptionSpec, OptionSpec, tuple[OptionLeg, OptionLeg]]:
    put_spec = OptionSpec(
        strike=float(put_quote["strike"]),
        time_to_expiry=time_to_expiry_years(
            entry_date=entry_date,
            expiry_date=expiry_date,
            quote_yte=put_quote.get("yte"),
            quote_dte=put_quote.get("dte"),
        ),
        option_type=OptionType.PUT,
    )
    call_spec = OptionSpec(
        strike=float(call_quote["strike"]),
        time_to_expiry=time_to_expiry_years(
            entry_date=entry_date,
            expiry_date=expiry_date,
            quote_yte=call_quote.get("yte"),
            quote_dte=call_quote.get("dte"),
        ),
        option_type=OptionType.CALL,
    )
    legs = (
        OptionLeg(
            spec=put_spec,
            entry_price=float(put_entry),
            side=PositionSide.SHORT,
            contract_multiplier=float(lot_size),
        ),
        OptionLeg(
            spec=call_spec,
            entry_price=float(call_entry),
            side=PositionSide.SHORT,
            contract_multiplier=float(lot_size),
        ),
    )
    return put_spec, call_spec, legs


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
    """Estimate margin per contract for one short-straddle position unit."""
    if margin_model is None:
        return None
    if not np.isfinite(spot) or spot <= 0:
        return None
    if not np.isfinite(volatility) or volatility <= 0:
        return None

    _, _, legs = _build_short_straddle_legs(
        entry_date=as_of_date,
        expiry_date=expiry_date,
        put_quote=put_quote,
        call_quote=call_quote,
        put_entry=put_entry,
        call_entry=call_entry,
        lot_size=lot_size,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))
    return margin_model.initial_margin_requirement(
        legs=legs,
        state=state,
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
    """Size short-straddle contracts from risk and margin budgets."""
    invalid_market = (
        not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(volatility)
        or volatility <= 0
    )
    if invalid_market:
        fallback = risk_sizer.min_contracts if risk_sizer else 1
        return fallback, None, None, None

    _, call_spec, legs = _build_short_straddle_legs(
        entry_date=entry_date,
        expiry_date=expiry_date,
        put_quote=put_quote,
        call_quote=call_quote,
        put_entry=put_entry,
        call_entry=call_entry,
        lot_size=lot_size,
    )
    state = MarketState(spot=float(spot), volatility=float(volatility))

    risk_contracts: int | None = None
    risk_per_contract: float | None = None
    risk_scenario: str | None = None
    if risk_sizer is not None:
        scenarios = scenario_generator.generate(spec=call_spec, state=state)
        if not scenarios:
            risk_contracts = risk_sizer.min_contracts
        else:
            risk_result = risk_estimator.estimate_risk_per_contract(
                legs=legs,
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
            legs=legs,
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
