"""Dynamic delta-hedging logic used during lifecycle mark valuation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ...config import ExecutionConfig
from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy
from .runtime_state import HedgeStepSnapshot


@dataclass(frozen=True, slots=True)
class HedgeExecutionResult:
    """Execution outcome for one hedge rebalance order."""

    fill_price: float
    total_cost: float


class HedgeTargetModel(Protocol):
    """Target model converting position state into a desired net delta."""

    def target_net_delta(
        self,
        *,
        policy: DeltaHedgePolicy,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        net_delta_before: float,
    ) -> float:
        """Return desired net delta after hedging."""


class HedgeExecutionModel(Protocol):
    """Execution model mapping hedge quantity into transaction costs/fills."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_price: float,
        execution: ExecutionConfig,
    ) -> HedgeExecutionResult:
        """Return execution result for one hedge rebalance trade."""


@dataclass(frozen=True, slots=True)
class DeltaNeutralHedgeTargetModel:
    """Baseline target model using policy target net delta as-is."""

    def target_net_delta(
        self,
        *,
        policy: DeltaHedgePolicy,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        net_delta_before: float,
    ) -> float:
        _ = (position, curr_date, option_delta, net_delta_before)
        return float(policy.target_net_delta)


@dataclass(frozen=True, slots=True)
class LinearHedgeExecutionModel:
    """Linear hedge execution model using configured slippage and commissions."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_price: float,
        execution: ExecutionConfig,
    ) -> HedgeExecutionResult:
        if trade_qty == 0.0:
            return HedgeExecutionResult(fill_price=float(hedge_price), total_cost=0.0)
        slippage = (
            execution.hedge.slip_ask if trade_qty > 0.0 else execution.hedge.slip_bid
        )
        fill_price = (
            float(hedge_price) + float(slippage)
            if trade_qty > 0.0
            else float(hedge_price) - float(slippage)
        )
        total_cost = abs(trade_qty) * (
            float(slippage) + execution.hedge.commission_per_unit
        )
        return HedgeExecutionResult(fill_price=fill_price, total_cost=float(total_cost))


class DeltaHedgeEngine:
    """Apply one-date delta hedging decisions for an open position."""

    def __init__(
        self,
        policy: DeltaHedgePolicy,
        *,
        target_model: HedgeTargetModel | None = None,
        execution_model: HedgeExecutionModel | None = None,
    ):
        self.policy = policy
        self.target_model = target_model or DeltaNeutralHedgeTargetModel()
        self.execution_model = execution_model or LinearHedgeExecutionModel()

    def apply(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        hedge_price: float,
        execution: ExecutionConfig,
    ) -> HedgeStepSnapshot:
        """Return one-date hedge snapshot for lifecycle valuation aggregation."""
        hedge_qty_before = float(position.hedge.qty)
        hedge_price_curr = float(hedge_price)
        net_delta_before = option_delta + hedge_qty_before
        target_net_delta = self.target_model.target_net_delta(
            policy=self.policy,
            position=position,
            curr_date=curr_date,
            option_delta=option_delta,
            net_delta_before=net_delta_before,
        )

        if not math.isfinite(hedge_price_curr):
            return HedgeStepSnapshot(
                hedge_price_prev=float(position.hedge.last_price),
                hedge_pnl=0.0,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
                trade_qty=0.0,
                hedge_cost=0.0,
            )

        hedge_price_prev = self._resolve_prev_hedge_price(
            position=position,
            hedge_price_curr=hedge_price_curr,
        )
        hedge_pnl = hedge_qty_before * (hedge_price_curr - hedge_price_prev)
        # Persist mark price each day so hedge carry is day-over-day.
        position.hedge.last_price = hedge_price_curr
        if not self._should_rebalance(
            curr_date=curr_date,
            net_delta_before=net_delta_before,
            target_net_delta=target_net_delta,
            position=position,
        ):
            return HedgeStepSnapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
                trade_qty=0.0,
                hedge_cost=0.0,
            )

        raw_trade_qty = target_net_delta - net_delta_before
        trade_qty = self._bounded_trade_qty(raw_trade_qty)
        if trade_qty == 0.0:
            return HedgeStepSnapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
                trade_qty=0.0,
                hedge_cost=0.0,
            )

        exec_result = self.execution_model.execute(
            trade_qty=trade_qty,
            hedge_price=hedge_price_curr,
            execution=execution,
        )
        trading_cost = exec_result.total_cost
        hedge_pnl -= trading_cost

        position.hedge.qty = hedge_qty_before + trade_qty
        position.hedge.last_rebalance_date = pd.Timestamp(curr_date)
        net_delta_after = option_delta + float(position.hedge.qty)
        return HedgeStepSnapshot(
            hedge_price_prev=hedge_price_prev,
            hedge_pnl=hedge_pnl,
            net_delta=net_delta_after,
            target_net_delta=target_net_delta,
            trade_qty=trade_qty,
            hedge_cost=trading_cost,
        )

    def _should_rebalance(
        self,
        *,
        curr_date: pd.Timestamp,
        net_delta_before: float,
        target_net_delta: float,
        position: OpenPosition,
    ) -> bool:
        if not self.policy.enabled:
            return False

        delta_trigger: bool | None = None
        if self.policy.trigger.delta_band_abs is not None:
            delta_gap = net_delta_before - target_net_delta
            delta_trigger = abs(delta_gap) >= self.policy.trigger.delta_band_abs

        time_trigger: bool | None = None
        if self.policy.trigger.rebalance_every_n_days is not None:
            last_date = position.hedge.last_rebalance_date
            if last_date is None:
                time_trigger = True
            else:
                elapsed_days = (
                    pd.Timestamp(curr_date).normalize()
                    - pd.Timestamp(last_date).normalize()
                ).days
                time_trigger = (
                    elapsed_days >= self.policy.trigger.rebalance_every_n_days
                )

        if delta_trigger is not None and time_trigger is not None:
            if self.policy.trigger.combine_mode == "and":
                return delta_trigger and time_trigger
            return delta_trigger or time_trigger
        if delta_trigger is not None:
            return delta_trigger
        if time_trigger is not None:
            return time_trigger
        return False

    def _bounded_trade_qty(self, trade_qty: float) -> float:
        bounded_qty = float(trade_qty)
        max_qty = self.policy.max_rebalance_qty
        if max_qty is not None:
            bounded_qty = max(min(bounded_qty, max_qty), -max_qty)
        if abs(bounded_qty) < self.policy.min_rebalance_qty:
            return 0.0
        return bounded_qty

    @staticmethod
    def _resolve_prev_hedge_price(
        *,
        position: OpenPosition,
        hedge_price_curr: float,
    ) -> float:
        hedge_prev = float(position.hedge.last_price)
        if math.isfinite(hedge_prev):
            return hedge_prev
        return hedge_price_curr
