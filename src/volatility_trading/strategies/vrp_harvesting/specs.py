"""VRP harvesting preset specification and factory.

This module maps a business-level VRP configuration into the generic
``StrategySpec`` contract consumed by the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from volatility_trading.backtesting.margin import MarginPolicy
from volatility_trading.backtesting.options_engine import (
    ExitRuleSet,
    LegSpec,
    SameDayReentryPolicy,
    StrategySpec,
    StructureSpec,
)
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


def _vrp_short_side(_leg_spec: LegSpec, _entry_direction: int) -> int:
    """VRP harvesting always sells structure legs."""
    return -1


# TODO: We may keep the args below only related to tehb stratgey istelf
# Things like reentry_policy may not be aprt of the strtagye in the sense
# thta it is somehtign we would play arround in order to make more money


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
    pricer: PriceModel = field(default_factory=BlackScholesPricer)
    scenario_generator: ScenarioGenerator = field(
        default_factory=FixedGridScenarioGenerator
    )
    risk_estimator: RiskEstimator = field(default_factory=StressLossRiskEstimator)
    risk_budget_pct: float | None = None
    margin_model: MarginModel | None = None
    margin_budget_pct: float | None = None
    margin_policy: MarginPolicy | None = None
    exit_rule_set: ExitRuleSet = field(default_factory=ExitRuleSet.period_rules)
    reentry_policy: SameDayReentryPolicy | None = None
    min_contracts: int = 1
    max_contracts: int | None = None

    def to_strategy_spec(self) -> StrategySpec:
        """Convert this VRP preset into the generic ``StrategySpec`` contract."""
        margin_model_resolved = self.margin_model
        if margin_model_resolved is None and self.margin_budget_pct is not None:
            margin_model_resolved = RegTMarginModel()

        margin_budget_pct_resolved = self.margin_budget_pct
        if margin_model_resolved is not None and margin_budget_pct_resolved is None:
            margin_budget_pct_resolved = 1.0

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
            rebalance_period=self.rebalance_period,
            max_holding_period=self.max_holding_period,
            exit_rule_set=self.exit_rule_set,
            reentry_policy=reentry_policy,
            pricer=self.pricer,
            scenario_generator=self.scenario_generator,
            risk_estimator=self.risk_estimator,
            risk_sizer=risk_sizer,
            margin_model=margin_model_resolved,
            margin_budget_pct=margin_budget_pct_resolved,
            margin_policy=self.margin_policy,
            min_contracts=self.min_contracts,
            max_contracts=self.max_contracts,
        )


def make_vrp_strategy(spec: VRPHarvestingSpec) -> StrategySpec:
    """Build an executable strategy specification from a VRP preset."""
    return spec.to_strategy_spec()
