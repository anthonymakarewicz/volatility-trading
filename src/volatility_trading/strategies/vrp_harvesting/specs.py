"""VRP harvesting preset specification and factory.

This module maps a business-level VRP configuration into the generic
``StrategySpec`` contract consumed by the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    ExitRuleSet,
    LegSpec,
    LifecycleConfig,
    SameDayReentryPolicy,
    SizingPolicyConfig,
    StrategySpec,
    StructureSpec,
)
from volatility_trading.filters import Filter
from volatility_trading.options import (
    OptionType,
    RiskBudgetSizer,
)
from volatility_trading.signals import Signal


def _vrp_short_side(_leg_spec: LegSpec, _entry_direction: int) -> int:
    """VRP harvesting always sells structure legs."""
    return -1


@dataclass
class VRPHarvestingSpec:
    """Configuration preset for baseline short-ATM-straddle VRP harvesting.

    The preset is intentionally small: it defines structure, sizing, margin, and
    reentry defaults, then delegates execution to shared options-engine components.
    """

    signal: Signal
    filters: tuple[Filter, ...] = ()
    rebalance_period: int | None = 5
    max_holding_period: int | None = None
    allow_same_day_reentry_on_rebalance: bool = True
    allow_same_day_reentry_on_max_holding: bool = False
    target_dte: int = 30
    max_dte_diff: int = 7
    risk_budget_pct: float | None = None
    margin_budget_pct: float | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy | None = None
    delta_hedge: DeltaHedgePolicy = field(default_factory=DeltaHedgePolicy)
    min_contracts: int = 1
    max_contracts: int | None = None

    def to_strategy_spec(self) -> StrategySpec:
        """Convert this VRP preset into the generic ``StrategySpec`` contract."""
        risk_sizer = (
            RiskBudgetSizer(
                risk_budget_pct=self.risk_budget_pct,
                min_contracts=self.min_contracts,
                max_contracts=self.max_contracts,
            )
            if self.risk_budget_pct is not None
            else None
        )
        reentry_policy = self.reentry_policy or SameDayReentryPolicy(
            allow_on_rebalance=self.allow_same_day_reentry_on_rebalance,
            allow_on_max_holding=self.allow_same_day_reentry_on_max_holding,
        )
        structure = StructureSpec(
            name="short_atm_straddle",
            dte_target=self.target_dte,
            dte_tolerance=self.max_dte_diff,
            legs=(
                LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                LegSpec(option_type=OptionType.CALL, delta_target=0.5),
            ),
        )
        return StrategySpec(
            name="vrp_harvesting",
            signal=self.signal,
            filters=self.filters,
            structure_spec=structure,
            side_resolver=_vrp_short_side,
            lifecycle=LifecycleConfig(
                rebalance_period=self.rebalance_period,
                max_holding_period=self.max_holding_period,
                exit_rule_set=self.exit_rule_set,
                reentry_policy=reentry_policy,
                delta_hedge=self.delta_hedge,
            ),
            sizing=SizingPolicyConfig(
                risk_sizer=risk_sizer,
                margin_budget_pct=self.margin_budget_pct,
                min_contracts=self.min_contracts,
                max_contracts=self.max_contracts,
            ),
        )


def make_vrp_strategy(spec: VRPHarvestingSpec) -> StrategySpec:
    """Build an executable strategy specification from a VRP preset."""
    return spec.to_strategy_spec()
