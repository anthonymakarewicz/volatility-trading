"""Private helpers shared by built-in strategy presets."""

from __future__ import annotations

from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    ExitRuleSet,
    LifecycleConfig,
    SameDayReentryPolicy,
    SizingPolicyConfig,
)
from volatility_trading.options import RiskBudgetSizer


def resolve_preset_reentry_policy(
    *,
    reentry_policy: SameDayReentryPolicy | None,
    allow_on_rebalance: bool,
    allow_on_max_holding: bool,
) -> SameDayReentryPolicy:
    """Return explicit reentry policy or build one from preset booleans."""
    return reentry_policy or SameDayReentryPolicy(
        allow_on_rebalance=allow_on_rebalance,
        allow_on_max_holding=allow_on_max_holding,
    )


def build_preset_lifecycle_config(
    *,
    rebalance_period: int | None,
    max_holding_period: int | None,
    exit_rule_set: ExitRuleSet,
    reentry_policy: SameDayReentryPolicy | None,
    allow_on_rebalance: bool,
    allow_on_max_holding: bool,
    delta_hedge: DeltaHedgePolicy,
) -> LifecycleConfig:
    """Build shared lifecycle config used by strategy presets."""
    return LifecycleConfig(
        rebalance_period=rebalance_period,
        max_holding_period=max_holding_period,
        exit_rule_set=exit_rule_set,
        reentry_policy=resolve_preset_reentry_policy(
            reentry_policy=reentry_policy,
            allow_on_rebalance=allow_on_rebalance,
            allow_on_max_holding=allow_on_max_holding,
        ),
        delta_hedge=delta_hedge,
    )


def build_preset_sizing_config(
    *,
    risk_budget_pct: float | None,
    margin_budget_pct: float | None,
    min_contracts: int,
    max_contracts: int | None,
) -> SizingPolicyConfig:
    """Build shared sizing config used by strategy presets."""
    risk_sizer = (
        RiskBudgetSizer(
            risk_budget_pct=risk_budget_pct,
            min_contracts=min_contracts,
            max_contracts=max_contracts,
        )
        if risk_budget_pct is not None
        else None
    )
    return SizingPolicyConfig(
        risk_sizer=risk_sizer,
        margin_budget_pct=margin_budget_pct,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
    )
