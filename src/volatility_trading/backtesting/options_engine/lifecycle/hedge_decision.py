"""Decision contracts and trigger logic for dynamic hedge rebalancing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..contracts.runtime import OpenPosition
from ..specs import DeltaHedgePolicy
from .hedge_policies import HedgeBandContext, evaluate_band_target


@dataclass(frozen=True, slots=True)
class HedgeDecision:
    """One-date hedge decision computed before any trade execution.

    ``center_target_net_delta`` is the frictionless center target from
    ``policy.target_net_delta``. ``target_net_delta`` may differ when policy chooses
    nearest-boundary behavior for no-trade bands.
    """

    center_target_net_delta: float
    target_net_delta: float
    delta_trigger: bool | None
    time_trigger: bool | None
    should_rebalance: bool
    band_half_width: float | None = None
    band_lower: float | None = None
    band_upper: float | None = None


class HedgeDecisionEngine:
    """Decision-only component for one-date hedge targeting and triggers."""

    def __init__(self, *, policy: DeltaHedgePolicy):
        self.policy = policy

    def decide(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        net_delta_before: float,
        band_context: HedgeBandContext,
    ) -> HedgeDecision:
        """Resolve center target, policy-adjusted target, and rebalance trigger."""
        center_target_net_delta = float(self.policy.target_net_delta)
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
        """Combine delta/time triggers using policy ``combine_mode`` semantics."""
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
        """Return periodic trigger state, or ``None`` when disabled."""
        if self.policy.trigger.rebalance_every_n_days is None:
            return None
        last_date = position.hedge.last_rebalance_date
        if last_date is None:
            # First eligible date can rebalance immediately when time trigger is used.
            return True
        elapsed_days = (
            pd.Timestamp(curr_date).normalize() - pd.Timestamp(last_date).normalize()
        ).days
        return elapsed_days >= self.policy.trigger.rebalance_every_n_days
