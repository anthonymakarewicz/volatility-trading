"""Shared open/mark/close lifecycle engine for arbitrary option structures."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from volatility_trading.options import MarginModel, PriceModel

from ...config import BacktestRunConfig
from ...data_contracts import HedgeMarketData
from ...margin import MarginPolicy
from ..contracts.records import MtmRecord
from ..contracts.runtime import LifecycleStepResult, OpenPosition, PositionEntrySetup
from ..exit_rules import ExitRuleSet
from ..specs import DeltaHedgePolicy
from .hedge_engine import HedgeExecutionModel
from .margining import evaluate_entry_margin
from .marking import build_mark_step_context, build_mark_step_snapshots
from .opening import build_open_position_state
from .option_execution import (
    OptionExecutionModel,
)
from .record_builders import build_entry_record
from .transitions import (
    transition_continue_open,
    transition_forced_liquidation,
    transition_standard_exit,
)
from .valuation import (
    entry_market_net_notional,
    entry_option_trade_cost,
    greeks_per_contract,
)

logger = logging.getLogger(__name__)


def _net_pnl_per_contract(
    *,
    position: OpenPosition,
    valuation_pnl_mtm: float,
    hedge_pnl: float,
) -> float | None:
    """Return unrealized net P&L per open contract for exit-rule evaluation."""
    contracts_open = int(position.contracts_open)
    if contracts_open <= 0:
        return None
    total_pnl = (
        float(valuation_pnl_mtm)
        - float(position.entry_option_trade_cost)
        + float(hedge_pnl)
    )
    return total_pnl / float(contracts_open)


@dataclass(frozen=True)
class PositionLifecycleEngine:
    """Shared position lifecycle logic: open, mark, margin-manage, and close."""

    rebalance_period: int | None
    max_holding_period: int | None
    exit_rule_set: ExitRuleSet
    margin_policy: MarginPolicy | None
    margin_model: MarginModel | None
    pricer: PriceModel
    delta_hedge_policy: DeltaHedgePolicy
    option_contract_multiplier: float
    hedge_market: HedgeMarketData | None = None
    hedge_execution_model: HedgeExecutionModel | None = None
    option_execution_model: OptionExecutionModel | None = None

    def open_position(
        self,
        *,
        setup: PositionEntrySetup,
        cfg: BacktestRunConfig,
        equity_running: float,
    ) -> tuple[OpenPosition, MtmRecord]:
        """Open one position and emit its entry-day MTM accounting record."""
        option_execution_model = (
            self.option_execution_model or cfg.execution.option_execution_model
        )
        if option_execution_model is None:
            raise ValueError("cfg.execution.option_execution_model must be configured")
        contracts_open = int(setup.contracts)
        option_contract_multiplier = float(self.option_contract_multiplier)
        entry_trade_cost = entry_option_trade_cost(
            legs=setup.intent.legs,
            option_contract_multiplier=option_contract_multiplier,
            contracts=contracts_open,
            option_execution_model=option_execution_model,
        )
        net_entry = entry_market_net_notional(
            legs=setup.intent.legs,
            option_contract_multiplier=option_contract_multiplier,
            contracts=contracts_open,
        )
        greeks_pc = greeks_per_contract(
            leg_quotes=tuple((leg, leg.quote) for leg in setup.intent.legs),
            option_contract_multiplier=option_contract_multiplier,
        )
        greeks = greeks_pc.scaled(contracts_open)
        net_delta = greeks.delta

        margin = evaluate_entry_margin(
            setup=setup,
            equity_running=equity_running,
            contracts_open=contracts_open,
            entry_option_trade_cost=entry_trade_cost,
            margin_policy=self.margin_policy,
        )
        entry_record = build_entry_record(
            setup=setup,
            contracts_open=contracts_open,
            greeks=greeks,
            net_delta=net_delta,
            margin=margin,
        )
        position = build_open_position_state(
            setup=setup,
            contracts_open=contracts_open,
            option_contract_multiplier=option_contract_multiplier,
            net_entry=net_entry,
            entry_option_trade_cost=entry_trade_cost,
            greeks=greeks,
            net_delta=net_delta,
            margin=margin,
            rebalance_period=self.rebalance_period,
            max_holding_period=self.max_holding_period,
        )
        return position, entry_record

    def mark_position(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestRunConfig,
        equity_running: float,
        exit_type_override: str | None = None,
    ) -> LifecycleStepResult:
        """Revalue one open position for one date and apply exit/liquidation rules.

        Returns:
            ``LifecycleStepResult`` containing updated position, MTM record,
            and emitted trade rows.
        """
        step = build_mark_step_context(
            curr_date=curr_date,
            cfg=cfg,
            equity_running=equity_running,
        )
        option_execution_model = (
            self.option_execution_model or cfg.execution.option_execution_model
        )
        if option_execution_model is None:
            raise ValueError("cfg.execution.option_execution_model must be configured")
        hedge_execution_model = (
            self.hedge_execution_model or cfg.execution.hedge_execution_model
        )
        if hedge_execution_model is None:
            raise ValueError("cfg.execution.hedge_execution_model must be configured")

        valuation, margin, mtm_record = build_mark_step_snapshots(
            position=position,
            step=step,
            options=options,
            margin_model=self.margin_model,
            pricer=self.pricer,
            delta_hedge_policy=self.delta_hedge_policy,
            hedge_market=self.hedge_market,
            hedge_execution_model=hedge_execution_model,
        )

        forced_outcome = transition_forced_liquidation(
            position=position,
            step=step,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
            margin_policy=self.margin_policy,
            option_execution_model=option_execution_model,
        )
        if forced_outcome is not None:
            return forced_outcome

        if valuation.has_missing_quote:
            logger.debug(
                "Missing quote on %s for position entry=%s; keeping position open",
                curr_date,
                position.entry_date,
            )
            return transition_continue_open(
                position=position,
                valuation=valuation,
                mtm_record=mtm_record,
            )

        pnl_per_contract = _net_pnl_per_contract(
            position=position,
            valuation_pnl_mtm=valuation.pnl_mtm,
            hedge_pnl=valuation.hedge.pnl,
        )
        exit_type = exit_type_override or self.exit_rule_set.evaluate(
            curr_date=step.curr_date,
            position=position,
            pnl_per_contract=pnl_per_contract,
        )
        if exit_type is None:
            return transition_continue_open(
                position=position,
                valuation=valuation,
                mtm_record=mtm_record,
            )

        return transition_standard_exit(
            position=position,
            step=step,
            exit_type=exit_type,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
            option_execution_model=option_execution_model,
        )
