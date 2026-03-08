"""Record builders for lifecycle MTM and trade outputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from volatility_trading.options.types import Greeks, MarketState

from ..contracts.market import QuoteSnapshot
from ..contracts.records import (
    MtmMargin,
    MtmRecord,
    TradeRecord,
)
from ..contracts.runtime import OpenPosition, PositionEntrySetup
from .runtime_state import (
    EntryMarginSnapshot,
    MarkMarginSnapshot,
    MarkValuationSnapshot,
)
from .valuation import trade_legs_payload


def build_entry_record(
    *,
    setup: PositionEntrySetup,
    contracts_open: int,
    greeks: Greeks,
    net_delta: float,
    margin: EntryMarginSnapshot,
) -> MtmRecord:
    """Build entry-day MTM record from entry and margin snapshots."""
    return MtmRecord(
        date=setup.intent.entry_date,
        market=MarketState(
            spot=(
                setup.intent.entry_state.spot
                if setup.intent.entry_state is not None
                else np.nan
            ),
            volatility=(
                setup.intent.entry_state.volatility
                if setup.intent.entry_state is not None
                else np.nan
            ),
        ),
        delta_pnl=margin.entry_delta_pnl,
        greeks=greeks,
        net_delta=net_delta,
        hedge_qty=0.0,
        hedge_price_prev=np.nan,
        hedge_pnl=0.0,
        option_market_pnl=0.0,
        option_trade_cost=margin.option_trade_cost,
        open_contracts=contracts_open,
        margin=MtmMargin(
            per_contract=margin.latest_margin_per_contract,
            initial_requirement=margin.initial_margin_requirement,
            core=margin.margin,
        ),
        hedge_carry_pnl=0.0,
        hedge_trade_cost=0.0,
        hedge_turnover=0.0,
        hedge_trade_count=0,
    )


def build_mark_record(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    valuation: MarkValuationSnapshot,
    margin: MarkMarginSnapshot,
) -> MtmRecord:
    """Build one-date MTM record before forced close or standard exits."""
    delta_pnl = (
        valuation.option_market_pnl + valuation.hedge.pnl + margin.margin.financing_pnl
    )
    return MtmRecord(
        date=curr_date,
        market=valuation.market,
        delta_pnl=delta_pnl,
        greeks=valuation.greeks,
        net_delta=valuation.net_delta,
        hedge_qty=position.hedge.qty,
        hedge_price_prev=valuation.hedge.price_prev,
        hedge_pnl=valuation.hedge.pnl,
        option_market_pnl=valuation.option_market_pnl,
        option_trade_cost=0.0,
        open_contracts=position.contracts_open,
        margin=MtmMargin(
            per_contract=position.latest_margin_per_contract,
            initial_requirement=margin.initial_margin_requirement,
            core=margin.margin,
        ),
        hedge_carry_pnl=valuation.hedge.telemetry.carry_pnl,
        hedge_trade_cost=valuation.hedge.telemetry.trade_cost,
        hedge_turnover=valuation.hedge.telemetry.turnover,
        hedge_trade_count=valuation.hedge.telemetry.trade_count,
    )


def build_trade_record(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    contracts: int,
    pnl: float,
    exit_type: str,
    exit_prices: tuple[float, ...],
    exit_leg_quotes: tuple[QuoteSnapshot, ...] | None = None,
    option_entry_cost: float = 0.0,
    option_exit_cost: float = 0.0,
) -> TradeRecord:
    """Build one typed trade ledger record for a close/liquidation action."""
    return TradeRecord(
        entry_date=position.entry_date,
        exit_date=curr_date,
        entry_dte=position.chosen_dte,
        expiry_date=position.expiry_date,
        contracts=contracts,
        pnl=pnl,
        risk_per_contract=position.risk_per_contract,
        risk_worst_scenario=position.risk_worst_scenario,
        margin_per_contract=position.latest_margin_per_contract,
        exit_type=exit_type,
        trade_legs=trade_legs_payload(
            legs=position.intent.legs,
            exit_prices=exit_prices,
            exit_leg_quotes=exit_leg_quotes,
        ),
        option_entry_cost=option_entry_cost,
        option_exit_cost=option_exit_cost,
    )


def apply_closed_position_fields(
    mtm_record: MtmRecord,
    *,
    delta_pnl: float,
    equity_after: float,
) -> MtmRecord:
    """Return MTM record with fields updated for a fully closed position."""
    return replace(
        mtm_record,
        delta_pnl=delta_pnl,
        greeks=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
        net_delta=0.0,
        hedge_qty=0.0,
        open_contracts=0,
        margin=replace(
            mtm_record.margin,
            initial_requirement=0.0,
            core=replace(
                mtm_record.margin.core,
                maintenance_margin_requirement=0.0,
                margin_excess=equity_after,
                margin_deficit=0.0,
                in_margin_call=False,
            ),
        ),
    )
