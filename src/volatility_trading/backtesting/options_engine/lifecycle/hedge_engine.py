"""Dynamic delta-hedging logic used during lifecycle mark valuation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ...config import ExecutionConfig
from ..contracts.market import HedgeMarketSnapshot
from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy
from .hedge_decision import (
    DeltaNeutralHedgeTargetModel,
    HedgeDecisionEngine,
    HedgeTargetModel,
)
from .hedge_execution import (
    FixedBpsExecutionModel,
    HedgeExecutionModel,
    HedgeExecutionResult,
    MidNoCostExecutionModel,
)
from .hedge_policies import HedgeBandContext
from .runtime_state import HedgeStepSnapshot, HedgeTelemetry, HedgeValuation

__all__ = [
    "DeltaHedgeEngine",
    "HedgeApplyContext",
    "HedgeTargetModel",
    "DeltaNeutralHedgeTargetModel",
    "HedgeExecutionResult",
    "HedgeExecutionModel",
    "MidNoCostExecutionModel",
    "FixedBpsExecutionModel",
]


@dataclass(frozen=True, slots=True)
class HedgeApplyContext:
    """One-date external inputs consumed by the hedge apply step."""

    curr_date: pd.Timestamp
    option_delta: float
    option_gamma: float
    option_volatility: float
    hedge_market: HedgeMarketSnapshot
    execution: ExecutionConfig


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
        self.decision_engine = HedgeDecisionEngine(
            policy=self.policy,
            target_model=self.target_model,
        )
        self.execution_model = execution_model or FixedBpsExecutionModel()

    def apply(
        self,
        *,
        position: OpenPosition,
        context: HedgeApplyContext,
    ) -> HedgeStepSnapshot:
        """Return one-date hedge snapshot for lifecycle valuation aggregation."""
        curr_date = pd.Timestamp(context.curr_date)
        option_delta = float(context.option_delta)
        hedge_market = context.hedge_market
        execution = context.execution
        hedge_qty_before = float(position.hedge.qty)
        hedge_price_curr = float(hedge_market.mid)
        hedge_delta_per_unit = float(hedge_market.contract_multiplier)
        if not math.isfinite(hedge_delta_per_unit) or hedge_delta_per_unit <= 0:
            raise ValueError("hedge_market.contract_multiplier must be finite and > 0")
        net_delta_before = option_delta + hedge_qty_before * hedge_delta_per_unit
        decision = self.decision_engine.decide(
            position=position,
            curr_date=curr_date,
            option_delta=option_delta,
            net_delta_before=net_delta_before,
            band_context=HedgeBandContext(
                option_gamma=float(context.option_gamma),
                option_volatility=float(context.option_volatility),
                hedge_price=hedge_price_curr,
                execution=execution,
            ),
        )

        if not math.isfinite(hedge_price_curr):
            return self._build_step_snapshot(
                hedge_price_prev=float(position.hedge.last_price),
                hedge_pnl=0.0,
                net_delta=net_delta_before,
                target_net_delta=decision.target_net_delta,
                trade_qty=0.0,
                trade_cost=0.0,
                carry_pnl=0.0,
                turnover=0.0,
                trade_count=0,
            )

        hedge_price_prev = self._resolve_prev_hedge_price(
            position=position,
            hedge_price_curr=hedge_price_curr,
        )
        hedge_carry_pnl = (
            hedge_qty_before
            * hedge_delta_per_unit
            * (hedge_price_curr - hedge_price_prev)
        )
        hedge_pnl = hedge_carry_pnl
        # Persist mark price each day so hedge carry is day-over-day.
        position.hedge.last_price = hedge_price_curr
        if not decision.should_rebalance:
            return self._build_step_snapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=decision.target_net_delta,
                trade_qty=0.0,
                trade_cost=0.0,
                carry_pnl=hedge_carry_pnl,
                turnover=0.0,
                trade_count=0,
            )

        raw_trade_qty = (
            decision.target_net_delta - net_delta_before
        ) / hedge_delta_per_unit
        trade_qty = self._bounded_trade_qty(raw_trade_qty)
        if trade_qty == 0.0:
            return self._build_step_snapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=decision.target_net_delta,
                trade_qty=0.0,
                trade_cost=0.0,
                carry_pnl=hedge_carry_pnl,
                turnover=0.0,
                trade_count=0,
            )

        exec_result = self.execution_model.execute(
            trade_qty=trade_qty,
            hedge_market=hedge_market,
            execution=execution,
        )
        trading_cost = exec_result.total_cost
        hedge_pnl -= trading_cost

        position.hedge.qty = hedge_qty_before + trade_qty
        position.hedge.last_rebalance_date = curr_date
        net_delta_after = (
            option_delta + float(position.hedge.qty) * hedge_delta_per_unit
        )
        return self._build_step_snapshot(
            hedge_price_prev=hedge_price_prev,
            hedge_pnl=hedge_pnl,
            net_delta=net_delta_after,
            target_net_delta=decision.target_net_delta,
            trade_qty=trade_qty,
            trade_cost=trading_cost,
            carry_pnl=hedge_carry_pnl,
            turnover=abs(trade_qty),
            trade_count=1,
        )

    @staticmethod
    def _build_step_snapshot(
        *,
        hedge_price_prev: float,
        hedge_pnl: float,
        net_delta: float,
        target_net_delta: float,
        trade_qty: float,
        trade_cost: float,
        carry_pnl: float,
        turnover: float,
        trade_count: int,
    ) -> HedgeStepSnapshot:
        return HedgeStepSnapshot(
            hedge=HedgeValuation(
                price_prev=hedge_price_prev,
                pnl=hedge_pnl,
                telemetry=HedgeTelemetry(
                    carry_pnl=carry_pnl,
                    trade_cost=trade_cost,
                    turnover=turnover,
                    trade_count=trade_count,
                ),
            ),
            net_delta=net_delta,
            target_net_delta=target_net_delta,
            trade_qty=trade_qty,
        )

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
