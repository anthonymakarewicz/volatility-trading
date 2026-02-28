"""Shared open/mark/close lifecycle engine for arbitrary option structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from volatility_trading.options import MarginModel, PriceModel

from ...config import BacktestRunConfig
from ...data_contracts import HedgeMarketData
from ...margin import MarginPolicy
from ..contracts.records import MtmRecord
from ..contracts.runtime import LifecycleStepResult, OpenPosition, PositionEntrySetup
from ..economics import roundtrip_commission_per_structure_contract
from ..exit_rules import ExitRuleSet
from ..specs import DeltaHedgePolicy
from .hedging import (
    DeltaNeutralHedgeTargetModel,
    HedgeExecutionModel,
    HedgeTargetModel,
    LinearHedgeExecutionModel,
)
from .margining import evaluate_entry_margin
from .marking import build_mark_step_context, build_mark_step_snapshots
from .opening import build_open_position_state
from .record_builders import build_entry_record
from .transitions import (
    transition_continue_open,
    transition_forced_liquidation,
    transition_standard_exit,
)
from .valuation import (
    entry_net_notional,
    greeks_per_contract,
)


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
    hedge_market: HedgeMarketData | None = None
    hedge_target_model: HedgeTargetModel = field(
        default_factory=DeltaNeutralHedgeTargetModel
    )
    hedge_execution_model: HedgeExecutionModel = field(
        default_factory=LinearHedgeExecutionModel
    )

    def open_position(
        self,
        *,
        setup: PositionEntrySetup,
        cfg: BacktestRunConfig,
        equity_running: float,
    ) -> tuple[OpenPosition, MtmRecord]:
        """Open one position and emit its entry-day MTM accounting record."""
        contracts_open = int(setup.contracts)
        lot_size = cfg.execution.lot_size
        roundtrip_commission_per_contract = roundtrip_commission_per_structure_contract(
            commission_per_leg=cfg.execution.commission_per_leg,
            legs=setup.intent.legs,
        )
        net_entry = entry_net_notional(
            legs=setup.intent.legs,
            lot_size=lot_size,
            contracts=contracts_open,
        )
        greeks_pc = greeks_per_contract(
            leg_quotes=tuple((leg, leg.quote) for leg in setup.intent.legs),
            lot_size=lot_size,
        )
        greeks = greeks_pc.scaled(contracts_open)
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
        position = build_open_position_state(
            setup=setup,
            contracts_open=contracts_open,
            net_entry=net_entry,
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
    ) -> LifecycleStepResult:
        """Revalue one open position for one date and apply exit/liquidation rules.

        Returns:
            ``LifecycleStepResult`` containing updated position, MTM record,
            and emitted trade rows.
        """
        step = build_mark_step_context(
            position=position,
            curr_date=curr_date,
            cfg=cfg,
            equity_running=equity_running,
        )

        valuation, margin, mtm_record = build_mark_step_snapshots(
            position=position,
            step=step,
            options=options,
            margin_model=self.margin_model,
            pricer=self.pricer,
            delta_hedge_policy=self.delta_hedge_policy,
            hedge_market=self.hedge_market,
            hedge_target_model=self.hedge_target_model,
            hedge_execution_model=self.hedge_execution_model,
        )

        forced_outcome = transition_forced_liquidation(
            position=position,
            step=step,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
            margin_policy=self.margin_policy,
        )
        if forced_outcome is not None:
            return forced_outcome

        if valuation.has_missing_quote:
            return transition_continue_open(
                position=position,
                valuation=valuation,
                mtm_record=mtm_record,
            )

        exit_type = self.exit_rule_set.evaluate(
            curr_date=step.curr_date, position=position
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
        )
