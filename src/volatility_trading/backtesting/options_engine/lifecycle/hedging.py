"""Dynamic delta-hedging logic used during lifecycle mark valuation."""

from __future__ import annotations

import math

import pandas as pd

from ...config import ExecutionConfig
from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy


class DeltaHedgeEngine:
    """Apply one-date delta hedging decisions for an open position."""

    def __init__(self, policy: DeltaHedgePolicy):
        self.policy = policy

    def apply(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        spot: float,
        execution: ExecutionConfig,
    ) -> tuple[float, float]:
        """Return daily hedge PnL and post-hedge net delta for one mark step."""
        hedge_qty_before = float(position.hedge_qty)
        spot_curr = float(spot)

        if not math.isfinite(spot_curr):
            net_delta = option_delta + hedge_qty_before
            return 0.0, net_delta

        spot_prev = self._resolve_prev_spot(position=position, spot_curr=spot_curr)
        hedge_pnl = hedge_qty_before * (spot_curr - spot_prev)

        net_delta_before = option_delta + hedge_qty_before
        if not self._should_rebalance(
            curr_date=curr_date,
            net_delta_before=net_delta_before,
            position=position,
        ):
            return hedge_pnl, net_delta_before

        raw_trade_qty = self.policy.target_net_delta - net_delta_before
        trade_qty = self._bounded_trade_qty(raw_trade_qty)
        if trade_qty == 0.0:
            return hedge_pnl, net_delta_before

        slippage = (
            execution.hedge_slip_ask if trade_qty > 0.0 else execution.hedge_slip_bid
        )
        trading_cost = abs(trade_qty) * (slippage + execution.hedge_commission_per_unit)
        hedge_pnl -= trading_cost

        position.hedge_qty = hedge_qty_before + trade_qty
        position.last_hedge_rebalance_date = pd.Timestamp(curr_date)
        net_delta_after = option_delta + float(position.hedge_qty)
        return hedge_pnl, net_delta_after

    def _should_rebalance(
        self,
        *,
        curr_date: pd.Timestamp,
        net_delta_before: float,
        position: OpenPosition,
    ) -> bool:
        if not self.policy.enabled:
            return False

        delta_trigger: bool | None = None
        if self.policy.delta_band_abs is not None:
            delta_gap = net_delta_before - self.policy.target_net_delta
            delta_trigger = abs(delta_gap) >= self.policy.delta_band_abs

        time_trigger: bool | None = None
        if self.policy.rebalance_every_n_days is not None:
            last_date = position.last_hedge_rebalance_date
            if last_date is None:
                time_trigger = True
            else:
                elapsed_days = (
                    pd.Timestamp(curr_date).normalize()
                    - pd.Timestamp(last_date).normalize()
                ).days
                time_trigger = elapsed_days >= self.policy.rebalance_every_n_days

        if delta_trigger is not None and time_trigger is not None:
            if self.policy.combine_mode == "and":
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
    def _resolve_prev_spot(*, position: OpenPosition, spot_curr: float) -> float:
        hedge_prev = float(position.hedge_price_entry)
        if math.isfinite(hedge_prev):
            return hedge_prev
        market_prev = float(position.last_market.spot)
        if math.isfinite(market_prev):
            return market_prev
        return spot_curr
