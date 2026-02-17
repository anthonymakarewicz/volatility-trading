from typing import Any, cast

import pytest

from volatility_trading.options import (
    FixedGridScenarioGenerator,
    MarketShock,
    MarketState,
    OptionLeg,
    OptionSpec,
    OptionType,
    PositionSide,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
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
