"""Dynamic delta-hedging logic used during lifecycle mark valuation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ...config import ExecutionConfig
from ..contracts.market import HedgeMarketSnapshot
from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy, FixedDeltaBandModel, WWDeltaBandModel
from .runtime_state import HedgeStepSnapshot, HedgeTelemetry, HedgeValuation


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
        hedge_market: HedgeMarketSnapshot,
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
        self.execution_model = execution_model or FixedBpsExecutionModel()

    def apply(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        option_gamma: float,
        option_volatility: float,
        hedge_market: HedgeMarketSnapshot,
        execution: ExecutionConfig,
    ) -> HedgeStepSnapshot:
        """Return one-date hedge snapshot for lifecycle valuation aggregation."""
        hedge_qty_before = float(position.hedge.qty)
        hedge_price_curr = float(hedge_market.mid)
        hedge_delta_per_unit = float(hedge_market.contract_multiplier)
        if not math.isfinite(hedge_delta_per_unit) or hedge_delta_per_unit <= 0:
            raise ValueError("hedge_market.contract_multiplier must be finite and > 0")
        net_delta_before = option_delta + hedge_qty_before * hedge_delta_per_unit
        center_target_net_delta = self.target_model.target_net_delta(
            policy=self.policy,
            position=position,
            curr_date=curr_date,
            option_delta=option_delta,
            net_delta_before=net_delta_before,
        )
        target_net_delta, delta_trigger = self._resolve_target_net_delta(
            center_target_net_delta=center_target_net_delta,
            net_delta_before=net_delta_before,
            option_gamma=option_gamma,
            option_volatility=option_volatility,
            hedge_price=hedge_price_curr,
            execution=execution,
        )

        if not math.isfinite(hedge_price_curr):
            return self._build_step_snapshot(
                hedge_price_prev=float(position.hedge.last_price),
                hedge_pnl=0.0,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
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
        if not self._should_rebalance(
            curr_date=curr_date,
            delta_trigger=delta_trigger,
            position=position,
        ):
            return self._build_step_snapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
                trade_qty=0.0,
                trade_cost=0.0,
                carry_pnl=hedge_carry_pnl,
                turnover=0.0,
                trade_count=0,
            )

        raw_trade_qty = (target_net_delta - net_delta_before) / hedge_delta_per_unit
        trade_qty = self._bounded_trade_qty(raw_trade_qty)
        if trade_qty == 0.0:
            return self._build_step_snapshot(
                hedge_price_prev=hedge_price_prev,
                hedge_pnl=hedge_pnl,
                net_delta=net_delta_before,
                target_net_delta=target_net_delta,
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
        position.hedge.last_rebalance_date = pd.Timestamp(curr_date)
        net_delta_after = (
            option_delta + float(position.hedge.qty) * hedge_delta_per_unit
        )
        return self._build_step_snapshot(
            hedge_price_prev=hedge_price_prev,
            hedge_pnl=hedge_pnl,
            net_delta=net_delta_after,
            target_net_delta=target_net_delta,
            trade_qty=trade_qty,
            trade_cost=trading_cost,
            carry_pnl=hedge_carry_pnl,
            turnover=abs(trade_qty),
            trade_count=1,
        )

    def _resolve_target_net_delta(
        self,
        *,
        center_target_net_delta: float,
        net_delta_before: float,
        option_gamma: float,
        option_volatility: float,
        hedge_price: float,
        execution: ExecutionConfig,
    ) -> tuple[float, bool | None]:
        band_model = self.policy.trigger.band_model
        center_target = float(center_target_net_delta)
        if band_model is None:
            return center_target, None

        band_half_width = self._resolve_band_half_width(
            band_model=band_model,
            option_gamma=option_gamma,
            option_volatility=option_volatility,
            hedge_price=hedge_price,
            execution=execution,
        )
        if not math.isfinite(band_half_width) or band_half_width < 0:
            return center_target, None

        lower = center_target - band_half_width
        upper = center_target + band_half_width

        if self.policy.rebalance_to == "nearest_boundary":
            if net_delta_before < lower:
                return lower, True
            if net_delta_before > upper:
                return upper, True
            return net_delta_before, False

        delta_trigger = not (lower <= net_delta_before <= upper)
        return center_target, delta_trigger

    @staticmethod
    def _resolve_band_half_width(
        *,
        band_model: FixedDeltaBandModel | WWDeltaBandModel,
        option_gamma: float,
        option_volatility: float,
        hedge_price: float,
        execution: ExecutionConfig,
    ) -> float:
        if isinstance(band_model, FixedDeltaBandModel):
            return float(band_model.half_width_abs)

        fee_bps = (
            band_model.fee_bps_override
            if band_model.fee_bps_override is not None
            else execution.hedge.fee_bps
        )
        fee_rate = float(fee_bps) / 10_000.0
        if not math.isfinite(fee_rate) or fee_rate <= 0:
            return 0.0

        gamma_eff = (
            max(abs(float(option_gamma)), band_model.gamma_floor)
            if math.isfinite(option_gamma)
            else float(band_model.gamma_floor)
        )
        sigma_eff = (
            max(float(option_volatility), band_model.sigma_floor)
            if math.isfinite(option_volatility)
            else float(band_model.sigma_floor)
        )
        spot_eff = (
            max(float(hedge_price), band_model.spot_floor)
            if math.isfinite(hedge_price)
            else float(band_model.spot_floor)
        )
        raw_band = band_model.calibration_c * (
            fee_rate / (gamma_eff * sigma_eff * spot_eff * spot_eff)
        ) ** (1.0 / 3.0)
        if not math.isfinite(raw_band):
            return float(band_model.max_band_abs)
        return float(
            max(
                band_model.min_band_abs,
                min(band_model.max_band_abs, raw_band),
            )
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

    def _should_rebalance(
        self,
        *,
        curr_date: pd.Timestamp,
        delta_trigger: bool | None,
        position: OpenPosition,
    ) -> bool:
        if not self.policy.enabled:
            return False

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
