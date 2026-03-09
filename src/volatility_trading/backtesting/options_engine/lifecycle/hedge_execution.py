"""Execution-model contracts and implementations for dynamic hedging.

Execution models map a desired hedge quantity into:
- a fill price assumption used for diagnostics, and
- an explicit trade cost deducted from hedge PnL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

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
    ) -> HedgeExecutionResult:
        """Return execution result for one hedge rebalance trade."""


@dataclass(frozen=True, slots=True)
class MidNoCostExecutionModel:
    """Baseline model: fill at mid and charge zero explicit trade cost."""

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
    ) -> HedgeExecutionResult:
        _ = trade_qty
        return HedgeExecutionResult(fill_price=float(hedge_market.mid), total_cost=0.0)


@dataclass(frozen=True, slots=True)
class FixedBpsExecutionModel:
    """Execution model using spread/slippage plus fixed-bps notional fee."""

    slip_ask: float = 0.0
    slip_bid: float = 0.0
    fee_bps: float = 1.0

    def __post_init__(self) -> None:
        if self.slip_ask < 0:
            raise ValueError("slip_ask must be >= 0")
        if self.slip_bid < 0:
            raise ValueError("slip_bid must be >= 0")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")

    def execute(
        self,
        *,
        trade_qty: float,
        hedge_market: HedgeMarketSnapshot,
    ) -> HedgeExecutionResult:
        """Map one hedge order into fill and explicit trading cost."""
        if trade_qty == 0.0:
            return HedgeExecutionResult(
                fill_price=float(hedge_market.mid), total_cost=0.0
            )
        # Reference the side-specific quote when available (buy->ask, sell->bid).
        reference_price = self._resolve_execution_reference_price(
            trade_qty=trade_qty,
            hedge_market=hedge_market,
        )
        slippage = float(self.slip_ask) if trade_qty > 0.0 else float(self.slip_bid)
        fill_price = (
            float(reference_price) + float(slippage)
            if trade_qty > 0.0
            else float(reference_price) - float(slippage)
        )
        # Cost is measured as distance from mid plus explicit bps fee on notional.
        price_cost_per_unit = abs(float(fill_price) - float(hedge_market.mid))
        fee_per_unit = abs(float(fill_price)) * (float(self.fee_bps) / 10_000.0)
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
