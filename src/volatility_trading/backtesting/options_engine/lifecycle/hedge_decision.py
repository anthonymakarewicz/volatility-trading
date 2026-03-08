"""Decision-model contracts and trigger logic for dynamic hedging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy
from .hedge_policies import HedgeBandContext, evaluate_band_target


@dataclass(frozen=True, slots=True)
class HedgeDecision:
    """One-date hedge decision computed before any trade execution."""

    center_target_net_delta: float
    target_net_delta: float
    delta_trigger: bool | None
    time_trigger: bool | None
    should_rebalance: bool
    band_half_width: float | None = None
    band_lower: float | None = None
    band_upper: float | None = None


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


class HedgeDecisionEngine:
    """Decision-only component for one-date hedge targeting and triggers."""

    def __init__(self, *, policy: DeltaHedgePolicy, target_model: HedgeTargetModel):
        self.policy = policy
        self.target_model = target_model

    def decide(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        option_delta: float,
        net_delta_before: float,
        band_context: HedgeBandContext,
    ) -> HedgeDecision:
        center_target_net_delta = self.target_model.target_net_delta(
            policy=self.policy,
            position=position,
            curr_date=curr_date,
            option_delta=option_delta,
            net_delta_before=net_delta_before,
        )
        band_decision = evaluate_band_target(
            policy=self.policy,
            center_target_net_delta=center_target_net_delta,
            net_delta_before=net_delta_before,
            context=band_context,
        )
        should_rebalance, time_trigger = self._resolve_rebalance_decision(
            curr_date=curr_date,
            delta_trigger=band_decision.delta_trigger,
            position=position,
        )
        return HedgeDecision(
            center_target_net_delta=float(center_target_net_delta),
            target_net_delta=float(band_decision.target_net_delta),
            delta_trigger=band_decision.delta_trigger,
            time_trigger=time_trigger,
            should_rebalance=should_rebalance,
            band_half_width=band_decision.band_half_width,
            band_lower=band_decision.band_lower,
            band_upper=band_decision.band_upper,
        )

    def _resolve_rebalance_decision(
        self,
        *,
        curr_date: pd.Timestamp,
        delta_trigger: bool | None,
        position: OpenPosition,
    ) -> tuple[bool, bool | None]:
        if not self.policy.enabled:
            return False, None

        time_trigger = self._resolve_time_trigger(
            curr_date=curr_date, position=position
        )
        if delta_trigger is not None and time_trigger is not None:
            if self.policy.trigger.combine_mode == "and":
                return delta_trigger and time_trigger, time_trigger
            return delta_trigger or time_trigger, time_trigger
        if delta_trigger is not None:
            return delta_trigger, time_trigger
        if time_trigger is not None:
            return time_trigger, time_trigger
        return False, time_trigger

    def _resolve_time_trigger(
        self, *, curr_date: pd.Timestamp, position: OpenPosition
    ) -> bool | None:
        if self.policy.trigger.rebalance_every_n_days is None:
            return None
        last_date = position.hedge.last_rebalance_date
        if last_date is None:
            return True
        elapsed_days = (
            pd.Timestamp(curr_date).normalize() - pd.Timestamp(last_date).normalize()
        ).days
        return elapsed_days >= self.policy.trigger.rebalance_every_n_days
