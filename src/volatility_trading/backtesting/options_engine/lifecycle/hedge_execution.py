"""Execution-model contracts and implementations for dynamic hedging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ...config import ExecutionConfig
from ..contracts.market import HedgeMarketSnapshot


@dataclass(frozen=True, slots=True)
class HedgeExecutionResult:
    """Execution outcome for one hedge rebalance order."""

    fill_price: float
    total_cost: float


class HedgeExecutionModel(Protocol):
    """Execution model mapping hedge quantity into transaction costs/fills."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
        execution: ExecutionConfig,
    ) -> HedgeExecutionResult:
        """Return execution result for one hedge rebalance trade."""


@dataclass(frozen=True, slots=True)
class MidNoCostExecutionModel:
    """Baseline hedge model filling at mid with zero explicit trade costs."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
        execution: ExecutionConfig,
    ) -> HedgeExecutionResult:
        _ = (trade_qty, execution)
        return HedgeExecutionResult(fill_price=float(hedge_market.mid), total_cost=0.0)


@dataclass(frozen=True, slots=True)
class FixedBpsExecutionModel:
    """Hedge execution model using spread/slippage plus fixed bps notional fees."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
        execution: ExecutionConfig,
    ) -> HedgeExecutionResult:
        if trade_qty == 0.0:
            return HedgeExecutionResult(
                fill_price=float(hedge_market.mid), total_cost=0.0
            )
        reference_price = self._resolve_execution_reference_price(
            trade_qty=trade_qty,
            hedge_market=hedge_market,
        )
        slippage = (
            execution.hedge.slip_ask if trade_qty > 0.0 else execution.hedge.slip_bid
        )
        fill_price = (
            float(reference_price) + float(slippage)
            if trade_qty > 0.0
            else float(reference_price) - float(slippage)
        )
        price_cost_per_unit = abs(float(fill_price) - float(hedge_market.mid))
        fee_per_unit = abs(float(fill_price)) * (execution.hedge.fee_bps / 10_000.0)
        total_cost = (
            abs(trade_qty)
            * float(hedge_market.contract_multiplier)
            * (price_cost_per_unit + fee_per_unit)
        )
        return HedgeExecutionResult(fill_price=fill_price, total_cost=float(total_cost))

    @staticmethod
    def _resolve_execution_reference_price(
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
    ) -> float:
        if trade_qty > 0.0 and math.isfinite(hedge_market.ask):
            return float(hedge_market.ask)
        if trade_qty < 0.0 and math.isfinite(hedge_market.bid):
            return float(hedge_market.bid)
        return float(hedge_market.mid)
