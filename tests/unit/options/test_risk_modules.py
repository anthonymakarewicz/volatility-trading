from typing import Any, cast

import pytest

from volatility_trading.options import (
    FixedGridScenarioGenerator,
    MarginModel,
    MarketShock,
    MarketState,
    OptionLeg,
    OptionSpec,
    OptionType,
    PortfolioMarginProxyModel,
    PositionSide,
    RegTMarginModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
    StressPoint,
    StressResult,
    StressScenario,
    contracts_for_risk_budget,
)


def test_fixed_grid_generator_builds_cartesian_grid():
    generator = FixedGridScenarioGenerator(
        spot_shocks_pct=(-0.10, 0.0, 0.10),
        vol_shocks=(0.0, 0.02),
        rate_shocks=(0.0,),
        time_shocks_years=(0.0, 1 / 365.0),
    )
    spec = OptionSpec(
        strike=100.0,
        time_to_expiry=30 / 365.0,
        option_type=OptionType.CALL,
    )
    state = MarketState(spot=100.0, volatility=0.2, rate=0.02, dividend_yield=0.01)

    scenarios = generator.generate(spec=spec, state=state)

    assert len(scenarios) == 12
    assert any(
        s.shock.d_spot == pytest.approx(10.0)
        and s.shock.d_volatility == pytest.approx(0.02)
        and s.shock.dt_years == pytest.approx(1 / 365.0)
        for s in scenarios
    )


def test_scenario_and_risk_estimators_are_protocol_compatible():
    assert isinstance(FixedGridScenarioGenerator(), ScenarioGenerator)
    assert isinstance(StressLossRiskEstimator(), RiskEstimator)


def test_margin_models_are_protocol_compatible():
    assert isinstance(RegTMarginModel(), MarginModel)
    assert isinstance(PortfolioMarginProxyModel(), MarginModel)


def test_worst_loss_estimator_for_short_straddle():
    class IntrinsicPricer:
        def price(self, spec: OptionSpec, state: MarketState) -> float:
            if spec.option_type == OptionType.CALL:
                return max(state.spot - spec.strike, 0.0)
            return max(spec.strike - state.spot, 0.0)

    strike = 100.0
    call_leg = OptionLeg(
        spec=OptionSpec(
            strike=strike,
            time_to_expiry=30 / 365.0,
            option_type=OptionType.CALL,
        ),
        entry_price=5.0,
        side=PositionSide.SHORT,
    )
    put_leg = OptionLeg(
        spec=OptionSpec(
            strike=strike,
            time_to_expiry=30 / 365.0,
            option_type=OptionType.PUT,
        ),
        entry_price=5.0,
        side=PositionSide.SHORT,
    )

    scenarios = (
        StressScenario(name="flat", shock=MarketShock(d_spot=0.0)),
        StressScenario(name="up", shock=MarketShock(d_spot=20.0)),
        StressScenario(name="down", shock=MarketShock(d_spot=-20.0)),
    )

    estimator = StressLossRiskEstimator()
    result = estimator.estimate_risk_per_contract(
        legs=[call_leg, put_leg],
        state=MarketState(spot=100.0, volatility=0.2, rate=0.0, dividend_yield=0.0),
        scenarios=scenarios,
        pricer=IntrinsicPricer(),
    )

    assert result.worst_loss == pytest.approx(10.0)
    assert result.worst_scenario.name in {"up", "down"}


def test_reg_t_margin_model_short_straddle_requirement():
    class ConstantPricer:
        def price(self, spec: OptionSpec, state: MarketState) -> float:
            _ = (spec, state)
            return 5.0

    state = MarketState(spot=100.0, volatility=0.2)
    legs = (
        OptionLeg(
            spec=OptionSpec(
                strike=100.0,
                time_to_expiry=30 / 365.0,
                option_type=OptionType.CALL,
            ),
            entry_price=5.0,
            side=PositionSide.SHORT,
            contract_multiplier=100.0,
        ),
        OptionLeg(
            spec=OptionSpec(
                strike=100.0,
                time_to_expiry=30 / 365.0,
                option_type=OptionType.PUT,
            ),
            entry_price=5.0,
            side=PositionSide.SHORT,
            contract_multiplier=100.0,
        ),
    )

    margin = RegTMarginModel().initial_margin_requirement(
        legs=legs,
        state=state,
        pricer=ConstantPricer(),
    )

    # For ATM straddle with constant option value:
    # short-side requirement = 500 + max(2000, 1000) = 2500
    # pair requirement       = greater_side + other_option_value = 2500 + 500
    assert margin == pytest.approx(3000.0)


def test_pm_proxy_margin_model_uses_stress_loss_and_house_overlay():
    class ConstantPricer:
        def price(self, spec: OptionSpec, state: MarketState) -> float:
            _ = (spec, state)
            return 2.0

    class OneScenarioGenerator:
        def generate(self, *, spec: OptionSpec, state: MarketState):
            _ = (spec, state)
            return (StressScenario(name="base", shock=MarketShock()),)

    class ConstantRiskEstimator:
        def estimate_risk_per_contract(self, *, legs, state, scenarios, pricer):
            _ = (legs, state, scenarios, pricer)
            base = StressScenario(name="base", shock=MarketShock())
            return StressResult(
                worst_loss=1000.0,
                worst_scenario=base,
                points=(StressPoint(scenario=base, pnl=-1000.0),),
            )

    model = PortfolioMarginProxyModel(
        scenario_generator=cast(ScenarioGenerator, OneScenarioGenerator()),
        risk_estimator=cast(RiskEstimator, ConstantRiskEstimator()),
        stress_multiplier=1.2,
        minimum_margin=800.0,
        house_multiplier=1.1,
        house_floor=500.0,
    )

    legs = (
        OptionLeg(
            spec=OptionSpec(
                strike=100.0,
                time_to_expiry=30 / 365.0,
                option_type=OptionType.CALL,
            ),
            entry_price=2.0,
            side=PositionSide.SHORT,
            contract_multiplier=100.0,
        ),
    )

    margin = model.initial_margin_requirement(
        legs=legs,
        state=MarketState(spot=100.0, volatility=0.2),
        pricer=ConstantPricer(),
    )

    assert margin == pytest.approx(1320.0)


def test_risk_estimator_clamps_invalid_shocked_inputs():
    class GuardPricer:
        def price(self, spec: OptionSpec, state: MarketState) -> float:
            assert spec.time_to_expiry > 0
            assert state.spot > 0
            assert state.volatility > 0
            return 1.0

    leg = OptionLeg(
        spec=OptionSpec(
            strike=100.0,
            time_to_expiry=5 / 365.0,
            option_type=OptionType.CALL,
        ),
        entry_price=1.0,
        side=PositionSide.LONG,
    )
    scenarios = (
        StressScenario(
            name="large_negative",
            shock=MarketShock(d_spot=-999.0, d_volatility=-1.0, dt_years=1.0),
        ),
    )

    estimator = StressLossRiskEstimator()
    result = estimator.estimate_risk_per_contract(
        legs=[leg],
        state=MarketState(spot=100.0, volatility=0.2),
        scenarios=scenarios,
        pricer=GuardPricer(),
    )

    assert result.worst_loss >= 0.0


def test_contract_sizing_uses_floor_and_caps():
    assert (
        contracts_for_risk_budget(
            equity=100_000.0,
            risk_budget_pct=0.02,
            risk_per_contract=350.0,
        )
        == 5
    )
    assert (
        contracts_for_risk_budget(
            equity=100_000.0,
            risk_budget_pct=0.02,
            risk_per_contract=350.0,
            max_contracts=4,
        )
        == 4
    )
    assert (
        contracts_for_risk_budget(
            equity=100_000.0,
            risk_budget_pct=0.02,
            risk_per_contract=0.0,
        )
        == 0
    )

    sizer = RiskBudgetSizer(risk_budget_pct=0.02, max_contracts=4)
    assert sizer.size(equity=100_000.0, risk_per_contract=350.0) == 4


def test_option_leg_side_requires_enum():
    leg_enum = OptionLeg(
        spec=OptionSpec(
            strike=100.0,
            time_to_expiry=30 / 365.0,
            option_type=OptionType.CALL,
        ),
        entry_price=2.0,
        side=PositionSide.SHORT,
    )
    assert leg_enum.side == PositionSide.SHORT

    with pytest.raises(
        ValueError, match="side must be PositionSide.SHORT or PositionSide.LONG"
    ):
        OptionLeg(
            spec=OptionSpec(
                strike=100.0,
                time_to_expiry=30 / 365.0,
                option_type=OptionType.PUT,
            ),
            entry_price=2.0,
            side=cast(Any, 1),
        )
