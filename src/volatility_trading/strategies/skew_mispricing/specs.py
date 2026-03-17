"""Skew mispricing preset specification and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    ExitRuleSet,
    LegSpec,
    LifecycleConfig,
    SameDayReentryPolicy,
    StrategySpec,
    StructureSpec,
)
from volatility_trading.filters import Filter
from volatility_trading.options import (
    OptionType,
)
from volatility_trading.signals import Signal, ZScoreSignal
from volatility_trading.signals._wrappers import InvertedSignal

from .._preset_helpers import (
    build_preset_lifecycle_config,
    build_preset_sizing_config,
    build_signal_driven_preset_exit_rule_set,
    resolve_preset_reentry_policy,
)
from .features import (
    build_skew_signal_input,
    resolve_orats_summary_tenor_suffix,
)


def _risk_reversal_side(leg_spec: LegSpec, entry_direction: int) -> int:
    """Resolve RR leg side from strategy direction.

    ``entry_direction=+1`` maps to long call / short put.
    ``entry_direction=-1`` maps to short call / long put.
    """
    if leg_spec.option_type == OptionType.CALL:
        return int(entry_direction)
    if leg_spec.option_type == OptionType.PUT:
        return -int(entry_direction)
    raise ValueError(
        "skew_mispricing risk reversal only supports call and put leg specs"
    )


def _default_signal() -> Signal:
    """Return the default contrarian z-score signal for raw skew."""
    return InvertedSignal(ZScoreSignal(window=60, entry=1.5, exit=0.5))


def _build_skew_signal_input_builder(*, target_dte: int):
    """Return one signal-input builder bound to a specific skew tenor."""
    return partial(build_skew_signal_input, target_dte=target_dte)


def _build_skew_lifecycle_config(spec: "SkewMispricingSpec") -> LifecycleConfig:
    """Build skew lifecycle, preferring signal-driven mode when no rebalance."""
    if spec.rebalance_period is None:
        return LifecycleConfig(
            rebalance_period=None,
            max_holding_period=spec.max_holding_period,
            exit_rule_set=build_signal_driven_preset_exit_rule_set(
                exit_rule_set=spec.exit_rule_set,
                max_holding_period=spec.max_holding_period,
                stop_loss_pnl_per_contract=spec.stop_loss_pnl_per_contract,
                take_profit_pnl_per_contract=spec.take_profit_pnl_per_contract,
            ),
            reentry_policy=resolve_preset_reentry_policy(
                reentry_policy=spec.reentry_policy,
                allow_on_rebalance=spec.allow_same_day_reentry_on_rebalance,
                allow_on_max_holding=spec.allow_same_day_reentry_on_max_holding,
                allow_on_stop_loss=spec.allow_same_day_reentry_on_stop_loss,
                allow_on_take_profit=spec.allow_same_day_reentry_on_take_profit,
            ),
            delta_hedge=spec.delta_hedge,
        )
    return build_preset_lifecycle_config(
        rebalance_period=spec.rebalance_period,
        max_holding_period=spec.max_holding_period,
        exit_rule_set=spec.exit_rule_set,
        reentry_policy=spec.reentry_policy,
        allow_on_rebalance=spec.allow_same_day_reentry_on_rebalance,
        allow_on_max_holding=spec.allow_same_day_reentry_on_max_holding,
        allow_on_stop_loss=spec.allow_same_day_reentry_on_stop_loss,
        allow_on_take_profit=spec.allow_same_day_reentry_on_take_profit,
        stop_loss_pnl_per_contract=spec.stop_loss_pnl_per_contract,
        take_profit_pnl_per_contract=spec.take_profit_pnl_per_contract,
        delta_hedge=spec.delta_hedge,
    )


@dataclass
class SkewMispricingSpec:
    """Configuration preset for 25-delta skew mean-reversion risk reversals."""

    signal: Signal = field(default_factory=_default_signal)
    filters: tuple[Filter, ...] = ()
    rebalance_period: int | None = None
    max_holding_period: int | None = 30
    allow_same_day_reentry_on_rebalance: bool = False
    allow_same_day_reentry_on_max_holding: bool = False
    allow_same_day_reentry_on_stop_loss: bool = False
    allow_same_day_reentry_on_take_profit: bool = False
    target_dte: int = 30
    max_dte_diff: int = 7
    delta_target_abs: float = 0.25
    delta_tolerance: float = 0.10
    risk_budget_pct: float | None = None
    margin_budget_pct: float | None = None
    stop_loss_pnl_per_contract: float | None = None
    take_profit_pnl_per_contract: float | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy | None = None
    delta_hedge: DeltaHedgePolicy = field(default_factory=DeltaHedgePolicy)
    min_contracts: int = 0
    max_contracts: int | None = None

    def __post_init__(self) -> None:
        resolve_orats_summary_tenor_suffix(self.target_dte)
        if self.max_dte_diff < 0:
            raise ValueError("max_dte_diff must be >= 0")
        if not 0 < self.delta_target_abs <= 1:
            raise ValueError("delta_target_abs must be in (0, 1]")
        if self.delta_tolerance <= 0:
            raise ValueError("delta_tolerance must be > 0")

    def to_strategy_spec(self) -> StrategySpec:
        """Convert this skew preset into the generic ``StrategySpec`` contract."""
        structure = StructureSpec(
            name="risk_reversal",
            dte_target=self.target_dte,
            dte_tolerance=self.max_dte_diff,
            legs=(
                LegSpec(
                    option_type=OptionType.PUT,
                    delta_target=-self.delta_target_abs,
                    delta_tolerance=self.delta_tolerance,
                ),
                LegSpec(
                    option_type=OptionType.CALL,
                    delta_target=self.delta_target_abs,
                    delta_tolerance=self.delta_tolerance,
                ),
            ),
        )
        return StrategySpec(
            name="skew_mispricing",
            signal=self.signal,
            filters=self.filters,
            signal_input_builder=_build_skew_signal_input_builder(
                target_dte=self.target_dte
            ),
            structure_spec=structure,
            side_resolver=_risk_reversal_side,
            lifecycle=_build_skew_lifecycle_config(self),
            sizing=build_preset_sizing_config(
                risk_budget_pct=self.risk_budget_pct,
                margin_budget_pct=self.margin_budget_pct,
                min_contracts=self.min_contracts,
                max_contracts=self.max_contracts,
            ),
        )


def make_skew_mispricing_strategy(spec: SkewMispricingSpec) -> StrategySpec:
    """Build an executable strategy specification from a skew preset."""
    return spec.to_strategy_spec()
