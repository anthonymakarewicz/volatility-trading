"""Typed strategy specification consumed by the backtesting engine."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import pandas as pd

from volatility_trading.filters import Filter
from volatility_trading.options import (
    PositionSide,
    RiskBudgetSizer,
)
from volatility_trading.signals import Signal

from ..data_contracts import HedgeMarketData
from .contracts.structures import LegSpec, StructureSpec
from .exit_rules import ExitRuleSet, MaxHoldingExitRule, SameDayReentryPolicy
from .factor_models import FactorDecompositionModel

SignalInput: TypeAlias = pd.Series | pd.DataFrame
SignalInputBuilder: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame | None, HedgeMarketData | None],
    SignalInput,
]
FilterContext: TypeAlias = pd.DataFrame | dict | pd.Series
FilterContextBuilder: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame | None, HedgeMarketData | None],
    FilterContext,
]
SideResolver: TypeAlias = Callable[[LegSpec, int], int | PositionSide]


def _default_signal_input(
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    hedge_market: HedgeMarketData | None,
) -> SignalInput:
    """Build default signal input when strategy does not customize it.

    The default returns a zero-valued Series indexed like the options panel.
    Concrete signals can ignore values and rely only on index cadence.
    """
    _ = (features, hedge_market)
    return pd.Series(0, index=options.index)


def _default_filter_context(
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    hedge_market: HedgeMarketData | None,
) -> FilterContext:
    """Build default filter context from features."""
    _ = hedge_market
    if features is not None:
        ctx = features.copy()
    else:
        ctx = pd.DataFrame(index=options.index.unique())
    return ctx


def _default_side_resolver(_leg: LegSpec, entry_direction: int) -> int:
    """Map strategy entry direction directly to each leg side."""
    return int(entry_direction)


@dataclass(frozen=True)
class FixedDeltaBandModel:
    """Static absolute half-width around the frictionless delta target."""

    half_width_abs: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_width_abs) or self.half_width_abs < 0:
            raise ValueError("half_width_abs must be >= 0")


@dataclass(frozen=True)
class WWDeltaBandModel:
    """Whalley-Wilmott-inspired dynamic delta no-trade band configuration."""

    calibration_c: float = 1.0
    min_band_abs: float = 0.0
    max_band_abs: float = 10.0
    gamma_floor: float = 1e-8
    sigma_floor: float = 1e-6
    spot_floor: float = 1e-8
    fee_bps_override: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.calibration_c) or self.calibration_c <= 0:
            raise ValueError("calibration_c must be > 0")
        if not math.isfinite(self.min_band_abs) or self.min_band_abs < 0:
            raise ValueError("min_band_abs must be >= 0")
        if not math.isfinite(self.max_band_abs) or self.max_band_abs <= 0:
            raise ValueError("max_band_abs must be > 0")
        if self.max_band_abs < self.min_band_abs:
            raise ValueError("max_band_abs must be >= min_band_abs")
        if not math.isfinite(self.gamma_floor) or self.gamma_floor <= 0:
            raise ValueError("gamma_floor must be > 0")
        if not math.isfinite(self.sigma_floor) or self.sigma_floor <= 0:
            raise ValueError("sigma_floor must be > 0")
        if not math.isfinite(self.spot_floor) or self.spot_floor <= 0:
            raise ValueError("spot_floor must be > 0")
        if self.fee_bps_override is not None and (
            not math.isfinite(self.fee_bps_override) or self.fee_bps_override < 0
        ):
            raise ValueError("fee_bps_override must be >= 0 when provided")


DeltaBandModel: TypeAlias = FixedDeltaBandModel | WWDeltaBandModel


@dataclass(frozen=True)
class HedgeTriggerPolicy:
    """Trigger policy deciding when to rebalance the dynamic hedge."""

    band_model: DeltaBandModel | None = None
    rebalance_every_n_days: int | None = None
    combine_mode: Literal["or", "and"] = "or"

    def __post_init__(self) -> None:
        if self.rebalance_every_n_days is not None and self.rebalance_every_n_days <= 0:
            raise ValueError("rebalance_every_n_days must be > 0 when provided")
        if self.combine_mode not in {"or", "and"}:
            raise ValueError("combine_mode must be either 'or' or 'and'")


@dataclass(frozen=True)
class DeltaHedgePolicy:
    """Dynamic delta-hedging policy applied during lifecycle mark steps."""

    enabled: bool = False
    target_net_delta: float = 0.0
    trigger: HedgeTriggerPolicy = field(default_factory=HedgeTriggerPolicy)
    rebalance_to: Literal["center", "nearest_boundary"] = "center"
    allow_missing_hedge_price: bool = False
    min_rebalance_qty: float = 0.0
    max_rebalance_qty: float | None = None

    def __post_init__(self) -> None:
        if self.min_rebalance_qty < 0:
            raise ValueError("min_rebalance_qty must be >= 0")
        if self.max_rebalance_qty is not None and self.max_rebalance_qty <= 0:
            raise ValueError("max_rebalance_qty must be > 0 when provided")
        if (
            self.max_rebalance_qty is not None
            and self.max_rebalance_qty < self.min_rebalance_qty
        ):
            raise ValueError("max_rebalance_qty must be >= min_rebalance_qty")
        if (
            self.enabled
            and self.trigger.band_model is None
            and self.trigger.rebalance_every_n_days is None
        ):
            raise ValueError(
                "enabled delta hedging requires trigger.band_model and/or "
                "rebalance_every_n_days"
            )
        if self.rebalance_to not in {"center", "nearest_boundary"}:
            raise ValueError("rebalance_to must be 'center' or 'nearest_boundary'")
        if (
            isinstance(self.trigger.band_model, WWDeltaBandModel)
            and self.rebalance_to != "nearest_boundary"
        ):
            raise ValueError(
                "WWDeltaBandModel requires rebalance_to='nearest_boundary'"
            )


@dataclass(frozen=True)
class LifecycleConfig:
    """Position lifecycle policy for one strategy.

    Periodic exits are optional. Strategies may run in a signal-driven mode
    with no rebalance or max-holding dates, optionally combined with a safety
    cap via `LifecycleConfig.signal_driven(max_holding_period=...)`.
    """

    rebalance_period: int | None = 5
    max_holding_period: int | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy = field(default_factory=SameDayReentryPolicy)
    delta_hedge: DeltaHedgePolicy = field(default_factory=DeltaHedgePolicy)

    @classmethod
    def signal_driven(
        cls,
        *,
        max_holding_period: int | None = None,
        reentry_policy: SameDayReentryPolicy | None = None,
        delta_hedge: DeltaHedgePolicy | None = None,
        exit_rule_set: ExitRuleSet | None = None,
    ) -> LifecycleConfig:
        """Build signal-driven lifecycle with optional max-holding safety cap."""
        resolved_exit_rule_set = exit_rule_set
        if resolved_exit_rule_set is None:
            resolved_exit_rule_set = (
                ExitRuleSet(rules=(MaxHoldingExitRule(),))
                if max_holding_period is not None
                else ExitRuleSet.no_rules()
            )
        return cls(
            rebalance_period=None,
            max_holding_period=max_holding_period,
            exit_rule_set=resolved_exit_rule_set,
            reentry_policy=(
                reentry_policy if reentry_policy is not None else SameDayReentryPolicy()
            ),
            delta_hedge=(
                delta_hedge if delta_hedge is not None else DeltaHedgePolicy()
            ),
        )

    def __post_init__(self) -> None:
        for name, period in (
            ("rebalance_period", self.rebalance_period),
            ("max_holding_period", self.max_holding_period),
        ):
            if period is not None and period <= 0:
                raise ValueError(f"{name} must be > 0 when provided")


@dataclass(frozen=True)
class SizingPolicyConfig:
    """Contract sizing policy for one strategy.

    `entry_risk_basis` controls whether risk-budget sizing uses the unhedged
    option package or the actual inception hedge package executed at entry.
    """

    risk_sizer: RiskBudgetSizer | None = None
    margin_budget_pct: float | None = None
    min_contracts: int = 1
    max_contracts: int | None = None
    entry_risk_basis: Literal["unhedged", "entry_hedged"] = "unhedged"

    def __post_init__(self) -> None:
        if self.min_contracts < 0:
            raise ValueError("min_contracts must be >= 0")
        if self.max_contracts is not None and self.max_contracts <= 0:
            raise ValueError("max_contracts must be > 0 when provided")
        if self.max_contracts is not None and self.max_contracts < self.min_contracts:
            raise ValueError("max_contracts must be >= min_contracts")
        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")
        if self.entry_risk_basis not in {"unhedged", "entry_hedged"}:
            raise ValueError("entry_risk_basis must be 'unhedged' or 'entry_hedged'")


@dataclass
class StrategySpec:
    """Full configuration contract consumed by the backtest engine.

    The spec bundles signal/filter plumbing, structure selection rules,
    lifecycle timing, and sizing policy so engine execution stays
    generic and strategy-agnostic.
    """

    signal: Signal
    structure_spec: StructureSpec
    name: str = "options_strategy"
    filters: tuple[Filter, ...] = ()
    signal_input_builder: SignalInputBuilder = _default_signal_input
    filter_context_builder: FilterContextBuilder = _default_filter_context
    side_resolver: SideResolver = _default_side_resolver

    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    sizing: SizingPolicyConfig = field(default_factory=SizingPolicyConfig)
    factor_decomposition_model: FactorDecompositionModel | None = None
