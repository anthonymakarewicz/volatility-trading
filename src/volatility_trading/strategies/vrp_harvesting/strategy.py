"""VRP preset built on top of the generic config-driven options strategy."""

from __future__ import annotations

from volatility_trading.backtesting import MarginPolicy
from volatility_trading.filters import Filter
from volatility_trading.options import (
    BlackScholesPricer,
    FixedGridScenarioGenerator,
    MarginModel,
    OptionType,
    PriceModel,
    RegTMarginModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
)
from volatility_trading.signals import Signal

from ..options_core import (
    ConfigDrivenOptionsStrategy,
    ExitRuleSet,
    LegSpec,
    OptionsStrategySpec,
    SameDayReentryPolicy,
    StructureSpec,
)


class VRPHarvestingStrategy(ConfigDrivenOptionsStrategy):
    """Baseline VRP harvesting strategy as a spec preset.

    The strategy remains a short-vol implementation by default (short ATM
    straddle), but all lifecycle plumbing is handled by
    `ConfigDrivenOptionsStrategy`.
    """

    @staticmethod
    def _entry_side_for_leg(_leg_spec: LegSpec, _entry_direction: int) -> int:
        return -1

    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        rebalance_period: int | None = 5,
        max_holding_period: int | None = None,
        allow_same_day_reentry_on_rebalance: bool = True,
        allow_same_day_reentry_on_max_holding: bool = False,
        target_dte: int = 30,
        max_dte_diff: int = 7,
        pricer: PriceModel | None = None,
        scenario_generator: ScenarioGenerator | None = None,
        risk_estimator: RiskEstimator | None = None,
        risk_budget_pct: float | None = None,
        margin_model: MarginModel | None = None,
        margin_budget_pct: float | None = None,
        margin_policy: MarginPolicy | None = None,
        exit_rule_set: ExitRuleSet | None = None,
        reentry_policy: SameDayReentryPolicy | None = None,
        min_contracts: int = 1,
        max_contracts: int | None = None,
    ):
        margin_model_resolved = margin_model
        if margin_model_resolved is None and margin_budget_pct is not None:
            margin_model_resolved = RegTMarginModel()

        margin_budget_pct_resolved = margin_budget_pct
        if margin_model_resolved is not None and margin_budget_pct_resolved is None:
            margin_budget_pct_resolved = 1.0

        risk_sizer = (
            RiskBudgetSizer(
                risk_budget_pct=risk_budget_pct,
                min_contracts=min_contracts,
                max_contracts=max_contracts,
            )
            if risk_budget_pct is not None
            else None
        )
        exit_rules = exit_rule_set or ExitRuleSet.period_rules()
        same_day_reentry = reentry_policy or SameDayReentryPolicy(
            allow_on_rebalance=allow_same_day_reentry_on_rebalance,
            allow_on_max_holding=allow_same_day_reentry_on_max_holding,
        )
        structure = StructureSpec(
            name="short_atm_straddle",
            dte_target=target_dte,
            dte_tolerance=max_dte_diff,
            legs=(
                LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                LegSpec(option_type=OptionType.CALL, delta_target=0.5),
            ),
        )
        spec = OptionsStrategySpec(
            name="VRPStrategy",
            signal=signal,
            filters=tuple(filters or ()),
            structure_spec=structure,
            side_resolver=self._entry_side_for_leg,
            rebalance_period=rebalance_period,
            max_holding_period=max_holding_period,
            exit_rule_set=exit_rules,
            reentry_policy=same_day_reentry,
            pricer=pricer or BlackScholesPricer(),
            scenario_generator=scenario_generator or FixedGridScenarioGenerator(),
            risk_estimator=risk_estimator or StressLossRiskEstimator(),
            risk_sizer=risk_sizer,
            margin_model=margin_model_resolved,
            margin_budget_pct=margin_budget_pct_resolved,
            margin_policy=margin_policy,
            min_contracts=min_contracts,
            max_contracts=max_contracts,
        )
        super().__init__(spec)

        # Keep strategy-level fields explicit for notebooks/reporting.
        self.target_dte = target_dte
        self.max_dte_diff = max_dte_diff
        self.allow_same_day_reentry_on_rebalance = allow_same_day_reentry_on_rebalance
        self.allow_same_day_reentry_on_max_holding = (
            allow_same_day_reentry_on_max_holding
        )
