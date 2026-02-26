"""Margin and financing helpers for options lifecycle execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount, MarginPolicy
from volatility_trading.backtesting.types import MarginCore
from volatility_trading.options import MarginModel, PriceModel

from ..sizing import estimate_entry_intent_margin_per_contract
from .state import (
    EntryMarginSnapshot,
    MarkMarginSnapshot,
    MarkValuationSnapshot,
    OpenPosition,
    PositionEntrySetup,
)
from .valuation import intent_with_updated_quotes


def evaluate_entry_margin(
    *,
    setup: PositionEntrySetup,
    equity_running: float,
    contracts_open: int,
    roundtrip_commission_per_contract: float,
    margin_policy: MarginPolicy | None,
) -> EntryMarginSnapshot:
    """Evaluate entry-day margin and financing state for one opened position."""
    margin_account = MarginAccount(margin_policy) if margin_policy is not None else None
    latest_margin_per_contract = setup.margin_per_contract
    initial_margin_requirement = (latest_margin_per_contract or 0.0) * contracts_open
    entry_delta_pnl = -(roundtrip_commission_per_contract * contracts_open)
    margin_core = MarginCore.empty()

    if margin_account is not None:
        margin_status = margin_account.evaluate(
            equity=equity_running + entry_delta_pnl,
            initial_margin_requirement=initial_margin_requirement,
            open_contracts=contracts_open,
            as_of=setup.intent.entry_date,
        )
        margin_core = margin_status.core
        entry_delta_pnl += margin_core.financing_pnl

    return EntryMarginSnapshot(
        margin_account=margin_account,
        latest_margin_per_contract=latest_margin_per_contract,
        initial_margin_requirement=initial_margin_requirement,
        entry_delta_pnl=entry_delta_pnl,
        margin=margin_core,
    )


def maybe_refresh_margin_per_contract(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    lot_size: int,
    valuation: MarkValuationSnapshot,
    margin_model: MarginModel | None,
    pricer: PriceModel,
) -> None:
    """Refresh per-contract margin estimate in-place when market state is valid."""
    if (
        margin_model is None
        or valuation.complete_leg_quotes is None
        or not np.isfinite(valuation.market.volatility)
        or valuation.market.volatility <= 0
    ):
        return

    current_intent = intent_with_updated_quotes(
        intent=position.intent,
        leg_quotes=valuation.complete_leg_quotes,
    )
    margin_pc_curr = estimate_entry_intent_margin_per_contract(
        intent=current_intent,
        as_of_date=curr_date,
        lot_size=lot_size,
        spot=float(valuation.market.spot),
        volatility=float(valuation.market.volatility),
        margin_model=margin_model,
        pricer=pricer,
    )
    if margin_pc_curr is not None:
        position.latest_margin_per_contract = margin_pc_curr


def evaluate_mark_margin(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    equity_running: float,
    valuation: MarkValuationSnapshot,
) -> MarkMarginSnapshot:
    """Evaluate one-date margin snapshot after MTM and before exits."""
    initial_margin_requirement = (
        position.latest_margin_per_contract or 0.0
    ) * position.contracts_open
    margin_core = MarginCore.empty()
    margin_status = None
    if position.margin_account is not None:
        margin_status = position.margin_account.evaluate(
            equity=equity_running + valuation.delta_pnl_market,
            initial_margin_requirement=initial_margin_requirement,
            open_contracts=position.contracts_open,
            as_of=curr_date,
        )
        margin_core = margin_status.core

    return MarkMarginSnapshot(
        initial_margin_requirement=initial_margin_requirement,
        margin=margin_core,
        margin_status=margin_status,
    )
