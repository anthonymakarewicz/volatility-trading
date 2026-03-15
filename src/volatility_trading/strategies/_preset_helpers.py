"""Private helpers shared by built-in strategy presets."""

from __future__ import annotations

from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    ExitRuleSet,
    LifecycleConfig,
    MaxHoldingExitRule,
    SameDayReentryPolicy,
    SizingPolicyConfig,
    StopLossExitRule,
    TakeProfitExitRule,
)
from volatility_trading.options import RiskBudgetSizer


def resolve_preset_reentry_policy(
    *,
    reentry_policy: SameDayReentryPolicy | None,
    allow_on_rebalance: bool,
    allow_on_max_holding: bool,
    allow_on_stop_loss: bool,
    allow_on_take_profit: bool,
) -> SameDayReentryPolicy:
    """Return explicit reentry policy or build one from preset booleans."""
    return reentry_policy or SameDayReentryPolicy(
        allow_on_rebalance=allow_on_rebalance,
        allow_on_max_holding=allow_on_max_holding,
        allow_on_stop_loss=allow_on_stop_loss,
        allow_on_take_profit=allow_on_take_profit,
    )


def build_preset_exit_rule_set(
    *,
    exit_rule_set: ExitRuleSet,
    stop_loss_pnl_per_contract: float | None,
    take_profit_pnl_per_contract: float | None,
) -> ExitRuleSet:
    """Append preset-owned richer exits to one base exit-rule set."""
    rules = list(exit_rule_set.rules)
    if stop_loss_pnl_per_contract is not None and any(
        isinstance(rule, StopLossExitRule) for rule in rules
    ):
        raise ValueError(
            "stop_loss_pnl_per_contract duplicates an explicit StopLossExitRule"
        )
    if take_profit_pnl_per_contract is not None and any(
        isinstance(rule, TakeProfitExitRule) for rule in rules
    ):
        raise ValueError(
            "take_profit_pnl_per_contract duplicates an explicit TakeProfitExitRule"
        )
    if stop_loss_pnl_per_contract is not None:
        rules.append(StopLossExitRule(stop_loss_pnl_per_contract))
    if take_profit_pnl_per_contract is not None:
        rules.append(TakeProfitExitRule(take_profit_pnl_per_contract))
    return ExitRuleSet(
        rules=tuple(rules),
        combined_rebalance_max_hold_type=exit_rule_set.combined_rebalance_max_hold_type,
    )


def build_signal_driven_preset_exit_rule_set(
    *,
    exit_rule_set: ExitRuleSet,
    max_holding_period: int | None,
    stop_loss_pnl_per_contract: float | None,
    take_profit_pnl_per_contract: float | None,
) -> ExitRuleSet:
    """Build signal-driven preset exits without reintroducing periodic rebalance.

    The default preset path should keep the signal-driven contract: no rebalance
    rule, optional max-holding safety cap, then any richer exit overlays.
    """
    base_exit_rule_set = exit_rule_set
    if exit_rule_set == ExitRuleSet.period_rules():
        base_exit_rule_set = ExitRuleSet(
            rules=(MaxHoldingExitRule(),) if max_holding_period is not None else ()
        )
    return build_preset_exit_rule_set(
        exit_rule_set=base_exit_rule_set,
        stop_loss_pnl_per_contract=stop_loss_pnl_per_contract,
        take_profit_pnl_per_contract=take_profit_pnl_per_contract,
    )


def build_preset_lifecycle_config(
    *,
    rebalance_period: int | None,
    max_holding_period: int | None,
    exit_rule_set: ExitRuleSet,
    reentry_policy: SameDayReentryPolicy | None,
    allow_on_rebalance: bool,
    allow_on_max_holding: bool,
    allow_on_stop_loss: bool,
    allow_on_take_profit: bool,
    stop_loss_pnl_per_contract: float | None,
    take_profit_pnl_per_contract: float | None,
    delta_hedge: DeltaHedgePolicy,
) -> LifecycleConfig:
    """Build shared lifecycle config used by strategy presets."""
    return LifecycleConfig(
        rebalance_period=rebalance_period,
        max_holding_period=max_holding_period,
        exit_rule_set=build_preset_exit_rule_set(
            exit_rule_set=exit_rule_set,
            stop_loss_pnl_per_contract=stop_loss_pnl_per_contract,
            take_profit_pnl_per_contract=take_profit_pnl_per_contract,
        ),
        reentry_policy=resolve_preset_reentry_policy(
            reentry_policy=reentry_policy,
            allow_on_rebalance=allow_on_rebalance,
            allow_on_max_holding=allow_on_max_holding,
            allow_on_stop_loss=allow_on_stop_loss,
            allow_on_take_profit=allow_on_take_profit,
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
