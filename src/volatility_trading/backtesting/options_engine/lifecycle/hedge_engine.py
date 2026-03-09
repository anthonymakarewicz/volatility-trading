"""Orchestrate one-date hedge decision, execution, and PnL accounting.

This module is the runtime bridge between:
- decision logic (target/trigger computation),
- execution logic (fill/cost assumptions), and
- lifecycle state mutation (hedge qty/mark timestamps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..contracts.market import HedgeMarketSnapshot
from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy
from .hedge_decision import (
    HedgeDecisionEngine,
)
from .hedge_execution import (
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    HedgeExecutionResult,
    MidNoCostHedgeExecutionModel,
)
from .hedge_policies import HedgeBandContext
from .runtime_state import HedgeStepSnapshot, HedgeTelemetry, HedgeValuation

__all__ = [
    "DeltaHedgeEngine",
    "HedgeApplyContext",
    "HedgeExecutionResult",
    "HedgeExecutionModel",
    "MidNoCostHedgeExecutionModel",
    "FixedBpsHedgeExecutionModel",
]


@dataclass(frozen=True, slots=True)
class HedgeApplyContext:
    """One-date external inputs consumed by ``DeltaHedgeEngine.apply``."""

    curr_date: pd.Timestamp
    option_delta: float
    option_gamma: float
    option_volatility: float
    hedge_market: HedgeMarketSnapshot


class DeltaHedgeEngine:
    """Apply one-date hedging updates to an open position state."""

    def __init__(
        self,
        policy: DeltaHedgePolicy,
        *,
        execution_model: HedgeExecutionModel | None = None,
    ):
        self.policy = policy
        self.decision_engine = HedgeDecisionEngine(policy=self.policy)
        self.execution_model = execution_model or FixedBpsHedgeExecutionModel()

    def apply(
        self,
        *,
        position: OpenPosition,
        context: HedgeApplyContext,
    ) -> HedgeStepSnapshot:
        """Evaluate and apply one hedge step.

        Side effects:
            - updates ``position.hedge.last_price`` each mark date when price is finite.
            - updates ``position.hedge.qty`` and ``last_rebalance_date`` when a trade
              is executed.
        """
        curr_date = pd.Timestamp(context.curr_date)
        option_delta = float(context.option_delta)
        hedge_market = context.hedge_market
        hedge_qty_before = float(position.hedge.qty)
        hedge_price_curr = float(hedge_market.mid)
        hedge_delta_per_unit = float(hedge_market.contract_multiplier)
        if not math.isfinite(hedge_delta_per_unit) or hedge_delta_per_unit <= 0:
            raise ValueError("hedge_market.contract_multiplier must be finite and > 0")
        net_delta_before = option_delta + hedge_qty_before * hedge_delta_per_unit
        decision = self.decision_engine.decide(
            position=position,
            curr_date=curr_date,
            net_delta_before=net_delta_before,
            band_context=HedgeBandContext(
                option_gamma=float(context.option_gamma),
                option_volatility=float(context.option_volatility),
                hedge_price=hedge_price_curr,
                hedge_fee_bps=self._resolve_execution_fee_bps(),
            ),
        )

        if not math.isfinite(hedge_price_curr):
            # Without a finite hedge mark, skip carry and trading but keep a snapshot.
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
        # Carry is realized on inventory held coming into the date (pre-trade qty).
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
        """Build canonical hedge step output consumed by record builders."""
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
        """Apply max-qty clipping and min-qty deadband to proposed trade size."""
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
        """Return previous mark price, defaulting to current when unavailable."""
        hedge_prev = float(position.hedge.last_price)
        if math.isfinite(hedge_prev):
            return hedge_prev
        return hedge_price_curr

    def _resolve_execution_fee_bps(self) -> float:
        """Read fee-bps from the configured execution model when available."""
        fee_bps = getattr(self.execution_model, "fee_bps", 0.0)
        try:
            fee_bps_value = float(fee_bps)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(fee_bps_value) or fee_bps_value < 0:
            return 0.0
        return fee_bps_value
