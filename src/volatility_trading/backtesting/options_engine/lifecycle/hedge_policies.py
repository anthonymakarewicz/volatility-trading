"""Runtime evaluators for hedge no-trade-band policies.

These helpers convert strategy policy settings into a per-date hedge
target/trigger decision, including fixed and Whalley-Wilmott-style bands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...config import ExecutionConfig
from ..specs import (
    DeltaBandModel,
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    WWDeltaBandModel,
)


@dataclass(frozen=True, slots=True)
class HedgeBandContext:
    """Runtime market/greek inputs used to evaluate a no-trade band."""

    option_gamma: float
    option_volatility: float
    hedge_price: float
    execution: ExecutionConfig


@dataclass(frozen=True, slots=True)
class HedgeBandDecision:
    """Target delta and trigger state produced by one band evaluation."""

    target_net_delta: float
    delta_trigger: bool | None
    band_half_width: float | None = None
    band_lower: float | None = None
    band_upper: float | None = None


def evaluate_band_target(
    *,
    policy: DeltaHedgePolicy,
    center_target_net_delta: float,
    net_delta_before: float,
    context: HedgeBandContext,
) -> HedgeBandDecision:
    """Evaluate one hedge target under the configured band policy.

    In ``center`` mode, target stays at the center and only trigger changes.
    In ``nearest_boundary`` mode, the target snaps to a boundary when outside
    the band and stays at current net delta when already inside.
    """
    band_model = policy.trigger.band_model
    center_target = float(center_target_net_delta)
    if band_model is None:
        return HedgeBandDecision(target_net_delta=center_target, delta_trigger=None)

    band_half_width = evaluate_band_half_width(band_model=band_model, context=context)
    if not math.isfinite(band_half_width) or band_half_width < 0:
        return HedgeBandDecision(target_net_delta=center_target, delta_trigger=None)

    lower = center_target - band_half_width
    upper = center_target + band_half_width

    if policy.rebalance_to == "nearest_boundary":
        if net_delta_before < lower:
            return HedgeBandDecision(
                target_net_delta=lower,
                delta_trigger=True,
                band_half_width=band_half_width,
                band_lower=lower,
                band_upper=upper,
            )
        if net_delta_before > upper:
            return HedgeBandDecision(
                target_net_delta=upper,
                delta_trigger=True,
                band_half_width=band_half_width,
                band_lower=lower,
                band_upper=upper,
            )
        # Inside the band: keep current net delta to avoid unnecessary recentering.
        return HedgeBandDecision(
            target_net_delta=float(net_delta_before),
            delta_trigger=False,
            band_half_width=band_half_width,
            band_lower=lower,
            band_upper=upper,
        )

    delta_trigger = not (lower <= net_delta_before <= upper)
    return HedgeBandDecision(
        target_net_delta=center_target,
        delta_trigger=delta_trigger,
        band_half_width=band_half_width,
        band_lower=lower,
        band_upper=upper,
    )


def evaluate_band_half_width(
    *, band_model: DeltaBandModel, context: HedgeBandContext
) -> float:
    """Evaluate absolute no-trade half-width from one concrete band model."""
    if isinstance(band_model, FixedDeltaBandModel):
        return float(band_model.half_width_abs)
    return _evaluate_ww_band_half_width(band_model=band_model, context=context)


def _evaluate_ww_band_half_width(
    *, band_model: WWDeltaBandModel, context: HedgeBandContext
) -> float:
    """Evaluate Whalley-Wilmott-style half-width with strict numeric guardrails."""
    fee_bps = (
        band_model.fee_bps_override
        if band_model.fee_bps_override is not None
        else _resolve_execution_fee_bps(context.execution)
    )
    fee_rate = float(fee_bps) / 10_000.0
    if not math.isfinite(fee_rate) or fee_rate <= 0:
        return 0.0

    # Floors keep the expression stable near expiry / tiny spot / tiny vol/gamma.
    gamma_eff = (
        max(abs(float(context.option_gamma)), band_model.gamma_floor)
        if math.isfinite(context.option_gamma)
        else float(band_model.gamma_floor)
    )
    sigma_eff = (
        max(float(context.option_volatility), band_model.sigma_floor)
        if math.isfinite(context.option_volatility)
        else float(band_model.sigma_floor)
    )
    spot_eff = (
        max(float(context.hedge_price), band_model.spot_floor)
        if math.isfinite(context.hedge_price)
        else float(band_model.spot_floor)
    )
    raw_band = band_model.calibration_c * (
        fee_rate / (gamma_eff * sigma_eff * spot_eff * spot_eff)
    ) ** (1.0 / 3.0)
    if not math.isfinite(raw_band):
        return float(band_model.max_band_abs)
    # Hard clamp keeps runtime behavior bounded and predictable.
    return float(
        max(
            band_model.min_band_abs,
            min(band_model.max_band_abs, raw_band),
        )
    )


def _resolve_execution_fee_bps(execution: ExecutionConfig) -> float:
    """Read hedge fee-bps from configured hedge execution model when available."""
    fee_bps = getattr(execution.hedge_execution_model, "fee_bps", 0.0)
    try:
        fee_bps_value = float(fee_bps)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fee_bps_value) or fee_bps_value < 0:
        return 0.0
    return fee_bps_value
