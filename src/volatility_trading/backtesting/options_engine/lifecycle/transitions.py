"""State transition handlers for mark/exit lifecycle steps."""

from __future__ import annotations

from dataclasses import replace

from volatility_trading.backtesting.margin import MarginPolicy

from ..contracts.records import MtmRecord, TradeRecord
from ..contracts.runtime import LifecycleStepResult, OpenPosition
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
    exit_prices_for_position,
    pnl_per_contract_from_exit_prices,
    update_position_mark_state,
)


def transition_forced_liquidation(
    *,
    position: OpenPosition,
    step: LifecycleStepContext,
    valuation: MarkValuationSnapshot,
    margin: MarkMarginSnapshot,
    mtm_record: MtmRecord,
    margin_policy: MarginPolicy | None,
) -> LifecycleStepResult | None:
    """Handle forced margin liquidation and return lifecycle outcome if triggered."""
    if (
        margin.margin_status is None
        or not margin.margin_status.core.forced_liquidation
        or margin.margin_status.contracts_to_liquidate <= 0
        or valuation.complete_leg_quotes is None
    ):
        return None

    exit_prices = exit_prices_for_position(
        position=position,
        leg_quotes=valuation.complete_leg_quotes,
        cfg=step.cfg,
    )
    contracts_to_close = margin.margin_status.contracts_to_liquidate
    contracts_after = position.contracts_open - contracts_to_close
    pnl_per_contract = pnl_per_contract_from_exit_prices(
        legs=position.intent.legs,
        exit_prices=exit_prices,
        lot_size=step.lot_size,
    )
    real_pnl_closed = pnl_per_contract * contracts_to_close + valuation.hedge_pnl
    pnl_net_closed = real_pnl_closed - (
        step.roundtrip_commission_per_contract * contracts_to_close
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
    )
    trade_rows: list[TradeRecord] = [trade_row]

    if contracts_after == 0:
        forced_delta_pnl = (
            pnl_net_closed - valuation.prev_mtm_before
        ) + margin.margin.financing_pnl
        equity_after = step.equity_running + forced_delta_pnl
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

    ratio_remaining = contracts_after / position.contracts_open
    pnl_mtm_remaining = valuation.pnl_mtm * ratio_remaining
    forced_delta_pnl = (
        pnl_net_closed + pnl_mtm_remaining - valuation.prev_mtm_before
    ) + margin.margin.financing_pnl
    greeks_remaining = valuation.greeks.scaled(ratio_remaining)
    net_delta_remaining = float(greeks_remaining.delta + position.hedge.qty)
    mtm_record = replace(
        mtm_record,
        delta_pnl=forced_delta_pnl,
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
) -> LifecycleStepResult:
    """Close a position on explicit exit rule trigger and build lifecycle outputs."""
    assert valuation.complete_leg_quotes is not None  # nosec B101
    exit_prices = exit_prices_for_position(
        position=position,
        leg_quotes=valuation.complete_leg_quotes,
        cfg=step.cfg,
    )
    pnl_per_contract = pnl_per_contract_from_exit_prices(
        legs=position.intent.legs,
        exit_prices=exit_prices,
        lot_size=step.lot_size,
    )
    real_pnl = pnl_per_contract * position.contracts_open + valuation.hedge_pnl
    pnl_net = real_pnl - (
        step.roundtrip_commission_per_contract * position.contracts_open
    )
    exit_delta_pnl = (pnl_net - valuation.prev_mtm_before) + margin.margin.financing_pnl
    equity_after = step.equity_running + exit_delta_pnl

    trade_row = build_trade_record(
        position=position,
        curr_date=step.curr_date,
        contracts=position.contracts_open,
        pnl=pnl_net,
        exit_type=exit_type,
        exit_prices=exit_prices,
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
