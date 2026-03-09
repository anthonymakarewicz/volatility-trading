"""State transition handlers for mark/exit lifecycle steps."""

from __future__ import annotations

import logging
from dataclasses import replace

from volatility_trading.backtesting.margin import MarginPolicy

from ..contracts.records import MtmRecord, TradeRecord
from ..contracts.runtime import LifecycleStepResult, OpenPosition
from .option_execution import OptionExecutionModel
from .record_builders import (
    apply_closed_position_fields,
    build_trade_record,
)
from .runtime_state import (
    LifecycleStepContext,
    MarkMarginSnapshot,
    MarkValuationSnapshot,
)
from .valuation import (
    execute_exit_for_position,
    update_position_mark_state,
)

logger = logging.getLogger(__name__)


def transition_forced_liquidation(
    *,
    position: OpenPosition,
    step: LifecycleStepContext,
    valuation: MarkValuationSnapshot,
    margin: MarkMarginSnapshot,
    mtm_record: MtmRecord,
    margin_policy: MarginPolicy | None,
    option_execution_model: OptionExecutionModel,
) -> LifecycleStepResult | None:
    """Handle forced margin liquidation and return lifecycle outcome if triggered."""
    if (
        margin.margin_status is None
        or not margin.margin_status.core.forced_liquidation
        or margin.margin_status.contracts_to_liquidate <= 0
        or valuation.complete_leg_quotes is None
    ):
        return None

    contracts_to_close = margin.margin_status.contracts_to_liquidate
    exit_prices, exit_trade_cost = execute_exit_for_position(
        position=position,
        leg_quotes=valuation.complete_leg_quotes,
        contracts_to_close=contracts_to_close,
        option_execution_model=option_execution_model,
    )
    contracts_after = position.contracts_open - contracts_to_close
    close_ratio = contracts_to_close / position.contracts_open
    entry_cost_closed = position.entry_option_trade_cost * close_ratio
    option_market_pnl_closed = valuation.pnl_mtm * close_ratio
    pnl_net_closed = (
        option_market_pnl_closed
        - entry_cost_closed
        - exit_trade_cost
        + valuation.hedge.pnl
    )

    trade_row = build_trade_record(
        position=position,
        curr_date=step.curr_date,
        contracts=contracts_to_close,
        pnl=pnl_net_closed,
        exit_type=(
            "Margin Call Liquidation"
            if contracts_after == 0
            else "Margin Call Partial Liquidation"
        ),
        exit_prices=exit_prices,
        exit_leg_quotes=valuation.complete_leg_quotes,
        option_entry_cost=entry_cost_closed,
        option_exit_cost=exit_trade_cost,
    )
    trade_rows: list[TradeRecord] = [trade_row]

    if contracts_after == 0:
        logger.warning(
            "Forced liquidation (full) date=%s entry_date=%s contracts_closed=%d pnl_net=%.6f",
            step.curr_date,
            position.entry_date,
            contracts_to_close,
            pnl_net_closed,
        )
        forced_delta_pnl = mtm_record.delta_pnl - exit_trade_cost
        equity_after = step.equity_running + forced_delta_pnl
        mtm_record = replace(
            mtm_record,
            option_trade_cost=exit_trade_cost,
        )
        mtm_record = apply_closed_position_fields(
            mtm_record,
            delta_pnl=forced_delta_pnl,
            equity_after=equity_after,
        )
        mtm_record = replace(
            mtm_record,
            margin=replace(
                mtm_record.margin,
                core=replace(
                    mtm_record.margin.core,
                    forced_liquidation=True,
                    contracts_liquidated=contracts_to_close,
                ),
            ),
        )
        return LifecycleStepResult(
            position=None,
            mtm_record=mtm_record,
            trade_rows=trade_rows,
        )

    logger.warning(
        (
            "Forced liquidation (partial) date=%s entry_date=%s "
            "contracts_closed=%d contracts_remaining=%d pnl_net_closed=%.6f"
        ),
        step.curr_date,
        position.entry_date,
        contracts_to_close,
        contracts_after,
        pnl_net_closed,
    )
    ratio_remaining = contracts_after / position.contracts_open
    pnl_mtm_remaining = valuation.pnl_mtm * ratio_remaining
    forced_delta_pnl = mtm_record.delta_pnl - exit_trade_cost
    greeks_remaining = valuation.greeks.scaled(ratio_remaining)
    net_delta_remaining = float(greeks_remaining.delta + position.hedge.qty)
    mtm_record = replace(
        mtm_record,
        delta_pnl=forced_delta_pnl,
        option_trade_cost=exit_trade_cost,
        greeks=greeks_remaining,
        net_delta=net_delta_remaining,
        open_contracts=contracts_after,
        margin=replace(
            mtm_record.margin,
            initial_requirement=(position.latest_margin_per_contract or 0.0)
            * contracts_after,
            core=replace(
                mtm_record.margin.core,
                maintenance_margin_requirement=(
                    position.latest_margin_per_contract or 0.0
                )
                * contracts_after
                * (
                    margin_policy.maintenance_margin_ratio
                    if margin_policy is not None
                    else 0.0
                ),
                contracts_liquidated=contracts_to_close,
            ),
        ),
    )
    position.contracts_open = contracts_after
    position.net_entry *= ratio_remaining
    position.entry_option_trade_cost *= ratio_remaining
    update_position_mark_state(
        position=position,
        pnl_mtm=pnl_mtm_remaining,
        market=valuation.market,
        greeks=greeks_remaining,
        net_delta=net_delta_remaining,
    )
    return LifecycleStepResult(
        position=position,
        mtm_record=mtm_record,
        trade_rows=trade_rows,
    )


def transition_standard_exit(
    *,
    position: OpenPosition,
    step: LifecycleStepContext,
    exit_type: str,
    valuation: MarkValuationSnapshot,
    margin: MarkMarginSnapshot,
    mtm_record: MtmRecord,
    option_execution_model: OptionExecutionModel,
) -> LifecycleStepResult:
    """Close a position on explicit exit rule trigger and build lifecycle outputs."""
    assert valuation.complete_leg_quotes is not None  # nosec B101
    exit_prices, exit_trade_cost = execute_exit_for_position(
        position=position,
        leg_quotes=valuation.complete_leg_quotes,
        contracts_to_close=position.contracts_open,
        option_execution_model=option_execution_model,
    )
    pnl_net = (
        valuation.pnl_mtm
        - position.entry_option_trade_cost
        - exit_trade_cost
        + valuation.hedge.pnl
    )
    exit_delta_pnl = mtm_record.delta_pnl - exit_trade_cost
    equity_after = step.equity_running + exit_delta_pnl

    trade_row = build_trade_record(
        position=position,
        curr_date=step.curr_date,
        contracts=position.contracts_open,
        pnl=pnl_net,
        exit_type=exit_type,
        exit_prices=exit_prices,
        exit_leg_quotes=valuation.complete_leg_quotes,
        option_entry_cost=position.entry_option_trade_cost,
        option_exit_cost=exit_trade_cost,
    )
    logger.info(
        (
            "Position exited date=%s entry_date=%s exit_type=%s contracts=%d "
            "pnl_net=%.6f exit_option_trade_cost=%.6f"
        ),
        step.curr_date,
        position.entry_date,
        exit_type,
        position.contracts_open,
        pnl_net,
        exit_trade_cost,
    )

    mtm_record = replace(
        mtm_record,
        option_trade_cost=exit_trade_cost,
    )
    mtm_record = apply_closed_position_fields(
        mtm_record,
        delta_pnl=exit_delta_pnl,
        equity_after=equity_after,
    )
    return LifecycleStepResult(
        position=None,
        mtm_record=mtm_record,
        trade_rows=[trade_row],
    )


def transition_continue_open(
    *,
    position: OpenPosition,
    valuation: MarkValuationSnapshot,
    mtm_record: MtmRecord,
) -> LifecycleStepResult:
    """Persist mark state and continue with an open position."""
    update_position_mark_state(
        position=position,
        pnl_mtm=valuation.pnl_mtm,
        market=valuation.market,
        greeks=mtm_record.greeks,
        net_delta=float(mtm_record.net_delta),
    )
    return LifecycleStepResult(
        position=position,
        mtm_record=mtm_record,
        trade_rows=[],
    )
