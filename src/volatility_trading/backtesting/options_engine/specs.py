"""Typed strategy specification consumed by the backtesting engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import pandas as pd

from volatility_trading.filters import Filter
from volatility_trading.options import (
    BlackScholesPricer,
    FixedGridScenarioGenerator,
    MarginModel,
    PositionSide,
    PriceModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
)
from volatility_trading.signals import Signal

from ..margin import MarginPolicy
from .exit_rules import ExitRuleSet, SameDayReentryPolicy
from .types import LegSpec, StructureSpec

SignalInput: TypeAlias = pd.Series | pd.DataFrame
SignalInputBuilder: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame | None, pd.Series | pd.DataFrame | None],
    SignalInput,
]
FilterContext: TypeAlias = pd.DataFrame | dict | pd.Series
FilterContextBuilder: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame | None, pd.Series | pd.DataFrame | None],
    FilterContext,
]
SideResolver: TypeAlias = Callable[[LegSpec, int], int | PositionSide]


def _default_signal_input(
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    hedge: pd.Series | pd.DataFrame | None,
) -> SignalInput:
    """Build default signal input when strategy does not customize it.

    The default returns a zero-valued Series indexed like the options panel.
    Concrete signals can ignore values and rely only on index cadence.
    """
    _ = (features, hedge)
    return pd.Series(0, index=options.index)


def _default_filter_context(
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    hedge: pd.Series | pd.DataFrame | None,
) -> FilterContext:
    """Build default filter context from features and optional hedge series."""
    if features is not None:
        ctx = features.copy()
    else:
        ctx = pd.DataFrame(index=options.index.unique())
    if hedge is None:
        return ctx
    if isinstance(hedge, pd.Series):
        hedge_name = hedge.name or "hedge"
        return ctx.join(hedge.rename(hedge_name), how="left")
    return ctx.join(hedge, how="left")


def _default_side_resolver(_leg: LegSpec, entry_direction: int) -> int:
    """Map strategy entry direction directly to each leg side."""
    return int(entry_direction)


@dataclass
class StrategySpec:
    """Full configuration contract consumed by the backtest engine.

    The spec bundles signal/filter plumbing, structure selection rules,
    lifecycle timing, and sizing/margin dependencies so engine execution stays
    generic and strategy-agnostic.
    """

    # TODO: Move generic backtest config into BacktestConfig

    signal: Signal
    structure_spec: StructureSpec
    name: str = "options_strategy"
    filters: tuple[Filter, ...] = ()
    signal_input_builder: SignalInputBuilder = _default_signal_input
    filter_context_builder: FilterContextBuilder = _default_filter_context
    side_resolver: SideResolver = _default_side_resolver

    options_data_key: str = "options"
    features_data_key: str = "features"
    hedge_data_key: str = "hedge"
    fallback_iv_feature_col: str = "iv_atm"

    rebalance_period: int | None = 5
    max_holding_period: int | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy = field(default_factory=SameDayReentryPolicy)

    pricer: PriceModel = field(default_factory=BlackScholesPricer)
    scenario_generator: ScenarioGenerator = field(
        default_factory=FixedGridScenarioGenerator
    )
    risk_estimator: RiskEstimator = field(default_factory=StressLossRiskEstimator)
    risk_sizer: RiskBudgetSizer | None = None

    margin_model: MarginModel | None = None
    margin_budget_pct: float | None = None
    margin_policy: MarginPolicy | None = None
    min_contracts: int = 1
    max_contracts: int | None = None

    def __post_init__(self) -> None:
        """Validate timing and sizing constraints at strategy construction time."""
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

        if self.min_contracts < 0:
            raise ValueError("min_contracts must be >= 0")
        if self.max_contracts is not None and self.max_contracts <= 0:
            raise ValueError("max_contracts must be > 0 when provided")
        if self.max_contracts is not None and self.max_contracts < self.min_contracts:
            raise ValueError("max_contracts must be >= min_contracts")

        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")
