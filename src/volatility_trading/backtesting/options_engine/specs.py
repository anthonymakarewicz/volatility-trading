"""Typed strategy specification consumed by the backtesting engine."""

from __future__ import annotations

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
from .exit_rules import ExitRuleSet, SameDayReentryPolicy

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
class HedgeTriggerPolicy:
    """Trigger policy deciding when to rebalance the dynamic hedge."""

    delta_band_abs: float | None = None
    rebalance_every_n_days: int | None = None
    combine_mode: Literal["or", "and"] = "or"

    def __post_init__(self) -> None:
        if self.delta_band_abs is not None and self.delta_band_abs < 0:
            raise ValueError("delta_band_abs must be >= 0 when provided")
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
            and self.trigger.delta_band_abs is None
            and self.trigger.rebalance_every_n_days is None
        ):
            raise ValueError(
                "enabled delta hedging requires delta_band_abs and/or "
                "rebalance_every_n_days"
            )


@dataclass(frozen=True)
class LifecycleConfig:
    """Position lifecycle policy for one strategy."""

    rebalance_period: int | None = 5
    max_holding_period: int | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy = field(default_factory=SameDayReentryPolicy)
    delta_hedge: DeltaHedgePolicy = field(default_factory=DeltaHedgePolicy)

    def __post_init__(self) -> None:
        for name, period in (
            ("rebalance_period", self.rebalance_period),
            ("max_holding_period", self.max_holding_period),
        ):
            if period is not None and period <= 0:
                raise ValueError(f"{name} must be > 0 when provided")
        if self.rebalance_period is None and self.max_holding_period is None:
            raise ValueError(
                "At least one of rebalance_period or max_holding_period must be set."
            )


@dataclass(frozen=True)
class SizingPolicyConfig:
    """Contract sizing policy for one strategy."""

    risk_sizer: RiskBudgetSizer | None = None
    margin_budget_pct: float | None = None
    min_contracts: int = 1
    max_contracts: int | None = None

    def __post_init__(self) -> None:
        if self.min_contracts < 0:
            raise ValueError("min_contracts must be >= 0")
        if self.max_contracts is not None and self.max_contracts <= 0:
            raise ValueError("max_contracts must be > 0 when provided")
        if self.max_contracts is not None and self.max_contracts < self.min_contracts:
            raise ValueError("max_contracts must be >= min_contracts")
        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")


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
