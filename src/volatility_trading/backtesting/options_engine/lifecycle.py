"""Shared open/mark/close lifecycle engine for arbitrary option structures."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from volatility_trading.options import Greeks, MarginModel, PriceModel

from ..margin import MarginPolicy
from ..types import BacktestConfig
from ._lifecycle.margining import (
    evaluate_entry_margin,
    evaluate_mark_margin,
    maybe_refresh_margin_per_contract,
)
from ._lifecycle.records import (
    apply_closed_position_fields,
    build_entry_record,
    build_mark_record,
    build_trade_row,
)
from ._lifecycle.state import (
    EntryMarginSnapshot,
    MarkMarginSnapshot,
    MarkValuationSnapshot,
    MtmRecord,
    OpenPosition,
    PositionEntrySetup,
)
from ._lifecycle.valuation import (
    entry_net_notional,
    exit_prices_for_position,
    greeks_per_contract,
    pnl_per_contract_from_exit_prices,
    resolve_mark_valuation,
    summary_expiry_and_dte_from_legs,
    update_position_mark_state,
)
from .exit_rules import ExitRuleSet


@dataclass(frozen=True)
class PositionLifecycleEngine:
    """Shared position lifecycle logic: open, mark, margin-manage, and close."""

    rebalance_period: int | None
    max_holding_period: int | None
    exit_rule_set: ExitRuleSet
    margin_policy: MarginPolicy | None
    margin_model: MarginModel | None
    pricer: PriceModel

    def _build_open_position(
        self,
        *,
        setup: PositionEntrySetup,
        contracts_open: int,
        net_entry: float,
        greeks: Greeks,
        net_delta: float,  # TODO: Remove it, keep a single delat that is updated with hedge instr.
        margin: EntryMarginSnapshot,
    ) -> OpenPosition:
        """Build mutable open-position state from entry lifecycle snapshots."""
        rebalance_date = (
            setup.intent.entry_date + pd.Timedelta(days=self.rebalance_period)
            if self.rebalance_period is not None
            else None
        )
        max_hold_date = (
            setup.intent.entry_date + pd.Timedelta(days=self.max_holding_period)
            if self.max_holding_period is not None
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
            risk_per_contract=setup.risk_per_contract,
            risk_worst_scenario=setup.risk_worst_scenario,
            margin_account=margin.margin_account,
            latest_margin_per_contract=margin.latest_margin_per_contract,
            net_entry=net_entry,
            prev_mtm=0.0,
            hedge_qty=0.0,
            hedge_price_entry=float("nan"),
            last_spot=(
                float(setup.intent.entry_state.spot)
                if setup.intent.entry_state is not None
                else float("nan")
            ),
            last_iv=(
                float(setup.intent.entry_state.volatility)
                if setup.intent.entry_state is not None
                else float("nan")
            ),
            last_delta=greeks.delta,
            last_gamma=greeks.gamma,
            last_vega=greeks.vega,
            last_theta=greeks.theta,
            last_net_delta=net_delta,
        )

    def _handle_forced_liquidation(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        cfg: BacktestConfig,
        lot_size: int,
        roundtrip_commission_per_contract: float,
        equity_running: float,
        valuation: MarkValuationSnapshot,
        margin: MarkMarginSnapshot,
        mtm_record: MtmRecord,
    ) -> tuple[OpenPosition | None, MtmRecord, list[dict]] | None:
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
            cfg=cfg,
        )
        contracts_to_close = margin.margin_status.contracts_to_liquidate
        contracts_after = position.contracts_open - contracts_to_close
        pnl_per_contract = pnl_per_contract_from_exit_prices(
            legs=position.intent.legs,
            exit_prices=exit_prices,
            lot_size=lot_size,
        )
        real_pnl_closed = pnl_per_contract * contracts_to_close + valuation.hedge_pnl
        pnl_net_closed = real_pnl_closed - (
            roundtrip_commission_per_contract * contracts_to_close
        )

        trade_row = build_trade_row(
            position=position,
            curr_date=curr_date,
            contracts=contracts_to_close,
            pnl=pnl_net_closed,
            exit_type=(
                "Margin Call Liquidation"
                if contracts_after == 0
                else "Margin Call Partial Liquidation"
            ),
            exit_prices=exit_prices,
        )
        trade_rows = [trade_row]

        if contracts_after == 0:
            forced_delta_pnl = (
                pnl_net_closed - valuation.prev_mtm_before
            ) + margin.margin.financing_pnl
            equity_after = equity_running + forced_delta_pnl
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
            return None, mtm_record, trade_rows

        ratio_remaining = contracts_after / position.contracts_open
        pnl_mtm_remaining = valuation.pnl_mtm * ratio_remaining
        forced_delta_pnl = (
            pnl_net_closed + pnl_mtm_remaining - valuation.prev_mtm_before
        ) + margin.margin.financing_pnl
        mtm_record = replace(
            mtm_record,
            delta_pnl=forced_delta_pnl,
            greeks=valuation.greeks.scaled(ratio_remaining),
            net_delta=valuation.net_delta * ratio_remaining,
            open_contracts=contracts_after,
            margin=replace(
                mtm_record.margin,
                initial_requirement=(
                    (position.latest_margin_per_contract or 0.0) * contracts_after
                ),
                core=replace(
                    mtm_record.margin.core,
                    maintenance_margin_requirement=(
                        (position.latest_margin_per_contract or 0.0)
                        * contracts_after
                        * (
                            self.margin_policy.maintenance_margin_ratio
                            if self.margin_policy is not None
                            else 0.0
                        )
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
            spot=valuation.spot,
            iv=valuation.iv,
            delta=float(mtm_record.greeks.delta),
            net_delta=float(mtm_record.net_delta),
            gamma=float(mtm_record.greeks.gamma),
            vega=float(mtm_record.greeks.vega),
            theta=float(mtm_record.greeks.theta),
        )
        return position, mtm_record, trade_rows

    def _handle_standard_exit(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        exit_type: str,
        cfg: BacktestConfig,
        lot_size: int,
        roundtrip_commission_per_contract: float,
        equity_running: float,
        valuation: MarkValuationSnapshot,
        margin: MarkMarginSnapshot,
        mtm_record: MtmRecord,
    ) -> tuple[OpenPosition | None, MtmRecord, list[dict]]:
        """Close a position on explicit exit rule trigger and build lifecycle outputs."""
        assert valuation.complete_leg_quotes is not None  # nosec B101
        exit_prices = exit_prices_for_position(
            position=position,
            leg_quotes=valuation.complete_leg_quotes,
            cfg=cfg,
        )
        pnl_per_contract = pnl_per_contract_from_exit_prices(
            legs=position.intent.legs,
            exit_prices=exit_prices,
            lot_size=lot_size,
        )
        real_pnl = pnl_per_contract * position.contracts_open + valuation.hedge_pnl
        pnl_net = real_pnl - (
            roundtrip_commission_per_contract * position.contracts_open
        )
        exit_delta_pnl = (
            pnl_net - valuation.prev_mtm_before
        ) + margin.margin.financing_pnl
        equity_after = equity_running + exit_delta_pnl

        trade_row = build_trade_row(
            position=position,
            curr_date=curr_date,
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
        return None, mtm_record, [trade_row]

    def open_position(
        self,
        *,
        setup: PositionEntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition, MtmRecord]:
        """Open one position and emit its entry-day MTM accounting record."""
        contracts_open = int(setup.contracts)
        lot_size = cfg.lot_size
        roundtrip_commission_per_contract = 2 * cfg.commission_per_leg
        net_entry = entry_net_notional(
            legs=setup.intent.legs,
            lot_size=lot_size,
            contracts=contracts_open,
        )
        delta_pc, gamma_pc, vega_pc, theta_pc = greeks_per_contract(
            leg_quotes=tuple((leg, leg.quote) for leg in setup.intent.legs),
            lot_size=lot_size,
        )  # TODO: Why not return the Greeks as object so that below we do greeks_pc * contracts_open ?
        greeks = Greeks(
            delta=delta_pc * contracts_open,
            gamma=gamma_pc * contracts_open,
            vega=vega_pc * contracts_open,
            theta=theta_pc * contracts_open,
        )
        net_delta = greeks.delta

        margin = evaluate_entry_margin(
            setup=setup,
            equity_running=equity_running,
            contracts_open=contracts_open,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
            margin_policy=self.margin_policy,
        )
        entry_record = build_entry_record(
            setup=setup,
            contracts_open=contracts_open,
            greeks=greeks,
            net_delta=net_delta,
            margin=margin,
        )
        position = self._build_open_position(
            setup=setup,
            contracts_open=contracts_open,
            net_entry=net_entry,
            greeks=greeks,
            net_delta=net_delta,
            margin=margin,
        )
        return position, entry_record

    def mark_position(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition | None, MtmRecord, list[dict]]:
        """Revalue one open position for one date and apply exit/liquidation rules.

        Returns:
            Tuple ``(updated_position_or_none, mtm_record, trade_rows)``.
        """
        lot_size = cfg.lot_size
        roundtrip_commission_per_contract = (
            2 * cfg.commission_per_leg
        )  # TODO: Scale it by nb of Options leg

        valuation = resolve_mark_valuation(
            position=position,
            curr_date=curr_date,
            options=options,
            lot_size=lot_size,
        )
        maybe_refresh_margin_per_contract(
            position=position,
            curr_date=curr_date,
            lot_size=lot_size,
            valuation=valuation,
            margin_model=self.margin_model,
            pricer=self.pricer,
        )
        margin = evaluate_mark_margin(
            position=position,
            curr_date=curr_date,
            equity_running=equity_running,
            valuation=valuation,
        )
        mtm_record = build_mark_record(
            position=position,
            curr_date=curr_date,
            valuation=valuation,
            margin=margin,
        )

        forced_outcome = self._handle_forced_liquidation(
            position=position,
            curr_date=curr_date,
            cfg=cfg,
            lot_size=lot_size,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
            equity_running=equity_running,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
        )
        if forced_outcome is not None:
            return forced_outcome

        if valuation.has_missing_quote:
            update_position_mark_state(
                position=position,
                pnl_mtm=valuation.pnl_mtm,
                spot=valuation.spot,
                iv=valuation.iv,
                delta=float(mtm_record.greeks.delta),
                net_delta=float(mtm_record.net_delta),
                gamma=float(mtm_record.greeks.gamma),
                vega=float(mtm_record.greeks.vega),
                theta=float(mtm_record.greeks.theta),
            )
            return position, mtm_record, []

        exit_type = self.exit_rule_set.evaluate(curr_date=curr_date, position=position)
        if exit_type is None:
            update_position_mark_state(  # TODO: The call is the same as above
                position=position,
                pnl_mtm=valuation.pnl_mtm,
                spot=valuation.spot,
                iv=valuation.iv,
                delta=float(mtm_record.greeks.delta),
                net_delta=float(mtm_record.net_delta),
                gamma=float(mtm_record.greeks.gamma),
                vega=float(mtm_record.greeks.vega),
                theta=float(mtm_record.greeks.theta),
            )
            return position, mtm_record, []

        return self._handle_standard_exit(
            position=position,
            curr_date=curr_date,
            exit_type=exit_type,
            cfg=cfg,
            lot_size=lot_size,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
            equity_running=equity_running,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
        )
