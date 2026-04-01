"""Open-position state builders for lifecycle entry transitions."""

from __future__ import annotations

import pandas as pd

from volatility_trading.options.types import Greeks, MarketState

from ..contracts.runtime import HedgeState, OpenPosition, PositionEntrySetup
from ..factor_models import FactorSnapshot
from .runtime_state import EntryMarginSnapshot
from .valuation import summary_expiry_and_dte_from_legs


def build_open_position_state(
    *,
    setup: PositionEntrySetup,
    contracts_open: int,
    option_contract_multiplier: float,
    net_entry: float,
    entry_option_trade_cost: float,
    greeks: Greeks,
    net_delta: float,
    factor_snapshot: FactorSnapshot,
    margin: EntryMarginSnapshot,
    hedge: HedgeState,
    cumulative_hedge_pnl: float,
    rebalance_period: int | None,
    max_holding_period: int | None,
) -> OpenPosition:
    """Build mutable open-position state from entry lifecycle snapshots."""
    rebalance_date = (
        setup.intent.entry_date + pd.Timedelta(days=rebalance_period)
        if rebalance_period is not None
        else None
    )
    max_hold_date = (
        setup.intent.entry_date + pd.Timedelta(days=max_holding_period)
        if max_holding_period is not None
        else None
    )
    expiry_summary, dte_summary = summary_expiry_and_dte_from_legs(
        entry_date=setup.intent.entry_date,
        legs=setup.intent.legs,
    )

    return OpenPosition(
        entry_date=setup.intent.entry_date,
        expiry_date=expiry_summary,
        chosen_dte=dte_summary,
        rebalance_date=rebalance_date,
        max_hold_date=max_hold_date,
        intent=setup.intent,
        contracts_open=contracts_open,
        option_contract_multiplier=float(option_contract_multiplier),
        risk_per_contract=setup.risk_per_contract,
        risk_worst_scenario=setup.risk_worst_scenario,
        entry_stress_points=setup.entry_stress_points,
        margin_account=margin.margin_account,
        latest_margin_per_contract=margin.latest_margin_per_contract,
        net_entry=net_entry,
        entry_option_trade_cost=entry_option_trade_cost,
        prev_mtm=0.0,
        hedge=hedge,
        last_market=(
            setup.intent.entry_state
            if setup.intent.entry_state is not None
            else MarketState(spot=float("nan"), volatility=float("nan"))
        ),
        last_greeks=greeks,
        last_net_delta=net_delta,
        last_factor_snapshot=factor_snapshot,
        cumulative_hedge_pnl=float(cumulative_hedge_pnl),
        cumulative_financing_pnl=margin.margin.financing_pnl,
        risk_budget_contracts=setup.risk_budget_contracts,
        margin_budget_contracts=setup.margin_budget_contracts,
        sizing_binding_constraint=setup.sizing_binding_constraint,
        min_contracts_override_applied=setup.min_contracts_override_applied,
    )
