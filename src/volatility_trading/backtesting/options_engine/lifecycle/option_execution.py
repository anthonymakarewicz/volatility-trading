"""Execution-model contracts and implementations for option-leg fills.

Option execution models map one leg order into:
- a fill price assumption used by entry/exit lifecycle paths, and
- an explicit trade cost that can be accounted separately from market PnL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ...config import ExecutionConfig
from ..contracts.market import QuoteSnapshot


@dataclass(frozen=True, slots=True)
class OptionExecutionOrder:
    """Inputs required to execute one option-leg order.

    Attributes:
        quote: Current quote snapshot for the leg.
        trade_side: Signed trade direction (`+1` buy, `-1` sell).
        quantity: Price-scaled quantity used for spread/slippage cost
            (`contracts * lot_size * abs(weight)` in current lifecycle accounting).
        fee_contracts: Contract count used for per-leg commission charging.
    """

    quote: QuoteSnapshot
    trade_side: int
    quantity: float
    fee_contracts: float

    def __post_init__(self) -> None:
        if self.trade_side not in (-1, 1):
            raise ValueError("trade_side must be -1 (sell) or +1 (buy)")
        if not math.isfinite(self.quantity) or self.quantity < 0:
            raise ValueError("quantity must be finite and >= 0")
        if not math.isfinite(self.fee_contracts) or self.fee_contracts < 0:
            raise ValueError("fee_contracts must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class OptionExecutionResult:
    """Execution outcome for one option-leg order."""

    fill_price: float
    total_cost: float
    price_cost: float
    fee_cost: float


class OptionExecutionModel(Protocol):
    """Execution model mapping one option-leg order into fill and costs."""

    def execute(
        self,
        *,
        order: OptionExecutionOrder,
        execution: ExecutionConfig,
    ) -> OptionExecutionResult:
        """Return execution result for one option-leg order."""


@dataclass(frozen=True, slots=True)
class MidNoCostOptionExecutionModel:
    """Baseline model: fill at mid and charge zero explicit trade cost."""

    def execute(
        self,
        *,
        order: OptionExecutionOrder,
        execution: ExecutionConfig,
    ) -> OptionExecutionResult:
        _ = execution
        mid = 0.5 * (float(order.quote.bid_price) + float(order.quote.ask_price))
        return OptionExecutionResult(
            fill_price=mid,
            total_cost=0.0,
            price_cost=0.0,
            fee_cost=0.0,
        )


@dataclass(frozen=True, slots=True)
class BidAskFeeOptionExecutionModel:
    """Execution model using bid/ask plus slippage and per-leg commissions."""

    def execute(
        self,
        *,
        order: OptionExecutionOrder,
        execution: ExecutionConfig,
    ) -> OptionExecutionResult:
        """Map one option-leg order into fill and explicit trade cost."""
        bid = float(order.quote.bid_price)
        ask = float(order.quote.ask_price)
        mid = 0.5 * (bid + ask)
        if order.quantity == 0.0 and order.fee_contracts == 0.0:
            return OptionExecutionResult(
                fill_price=mid,
                total_cost=0.0,
                price_cost=0.0,
                fee_cost=0.0,
            )

        reference_price = self._resolve_reference_price(
            trade_side=order.trade_side,
            bid=bid,
            ask=ask,
            mid=mid,
        )
        slippage = (
            float(execution.slip_ask)
            if order.trade_side > 0
            else float(execution.slip_bid)
        )
        fill_price = (
            float(reference_price) + slippage
            if order.trade_side > 0
            else float(reference_price) - slippage
        )
        price_cost = abs(fill_price - mid) * float(order.quantity)
        fee_cost = float(execution.commission_per_leg) * float(order.fee_contracts)
        total_cost = price_cost + fee_cost
        return OptionExecutionResult(
            fill_price=float(fill_price),
            total_cost=float(total_cost),
            price_cost=float(price_cost),
            fee_cost=float(fee_cost),
        )

    @staticmethod
    def _resolve_reference_price(
        *,
        trade_side: int,
        bid: float,
        ask: float,
        mid: float,
    ) -> float:
        if trade_side > 0 and math.isfinite(ask):
            return ask
        if trade_side < 0 and math.isfinite(bid):
            return bid
        return mid
