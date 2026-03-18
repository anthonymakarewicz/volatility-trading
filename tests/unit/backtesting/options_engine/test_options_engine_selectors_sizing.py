import pandas as pd
import pytest

from volatility_trading.backtesting.options_engine import LegSpec
from volatility_trading.backtesting.options_engine.contracts.market import QuoteSnapshot
from volatility_trading.backtesting.options_engine.contracts.structures import (
    EntryIntent,
    LegSelection,
)
from volatility_trading.backtesting.options_engine.selectors import (
    select_best_expiry_for_leg_group,
)
from volatility_trading.backtesting.options_engine.sizing import (
    SizingRequest,
    estimate_entry_intent_margin_per_contract,
    size_entry_intent,
)
from volatility_trading.options import (
    MarketShock,
    OptionType,
    RiskBudgetSizer,
    StressPoint,
    StressResult,
    StressScenario,
)


def _quote(
    option_type: str,
    *,
    strike: float = 100.0,
    dte: int = 30,
    expiry_date: str = "2020-01-31",
    include_yte: bool = True,
) -> QuoteSnapshot:
    payload = {
        "option_type": option_type,
        "strike": strike,
        "dte": dte,
        "expiry_date": pd.Timestamp(expiry_date),
        "delta": -0.5 if option_type == "P" else 0.5,
        "bid_price": 5.0,
        "ask_price": 5.2,
    }
    if include_yte:
        payload["yte"] = dte / 365.0
    return QuoteSnapshot.from_series(pd.Series(payload))


def _short_straddle_intent() -> EntryIntent:
    put = _quote("P")
    call = _quote("C")
    return EntryIntent(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(
            LegSelection(
                spec=LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                quote=put,
                side=-1,
                entry_price=5.0,
            ),
            LegSelection(
                spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
                quote=call,
                side=-1,
                entry_price=5.0,
            ),
        ),
    )


def _long_call_intent() -> EntryIntent:
    call = _quote("C")
    return EntryIntent(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(
            LegSelection(
                spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
                quote=call,
                side=1,
                entry_price=5.0,
            ),
        ),
    )


def test_select_best_expiry_for_leg_group_prefers_lower_combined_score():
    chain = pd.DataFrame(
        {
            "expiry_date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-02",
                    "2020-02-02",
                ]
            ),
            "dte": [29, 29, 31, 31],
            "option_type": ["P", "C", "P", "C"],
            "delta": [-0.58, 0.58, -0.51, 0.51],
            "strike": [100.0, 100.0, 100.0, 100.0],
            "bid_price": [5.0, 5.0, 5.0, 5.0],
            "ask_price": [6.4, 6.4, 5.3, 5.3],
            "open_interest": [100, 100, 100, 100],
            "volume": [100, 100, 100, 100],
            "spot_price": [100.0, 100.0, 100.0, 100.0],
        }
    )

    out = select_best_expiry_for_leg_group(
        chain=chain,
        group_legs=(
            LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
            LegSpec(option_type=OptionType.CALL, delta_target=0.5),
        ),
        target_dte=30,
        dte_tolerance=2,
    )

    assert out is not None
    expiry, chosen_dte, quotes = out
    assert expiry == pd.Timestamp("2020-02-02")
    assert chosen_dte == 31
    assert len(quotes) == 2
    assert all(isinstance(q, QuoteSnapshot) for q in quotes)


def test_size_entry_intent_uses_risk_and_margin_caps():
    class OneScenarioGenerator:
        def generate(self, *, spec, state):
            _ = (spec, state)
            return (StressScenario(name="base", shock=MarketShock()),)

    class ConstantRiskEstimator:
        def estimate_risk_per_contract(self, *, legs, state, scenarios, pricer):
            _ = (legs, state, scenarios, pricer)
            base = StressScenario(name="base", shock=MarketShock())
            return StressResult(
                worst_loss=250.0,
                worst_scenario=base,
                points=(StressPoint(scenario=base, pnl=-250.0),),
            )

    class ConstantMarginModel:
        def initial_margin_requirement(self, *, legs, state, pricer):
            _ = (legs, state, pricer)
            return 500.0

    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    decision = size_entry_intent(
        SizingRequest(
            intent=_short_straddle_intent(),
            option_contract_multiplier=1,
            spot=100.0,
            volatility=0.2,
            equity=10_000.0,
            pricer=NullPricer(),
            scenario_generator=OneScenarioGenerator(),
            risk_estimator=ConstantRiskEstimator(),
            risk_sizer=RiskBudgetSizer(risk_budget_pct=0.10, min_contracts=0),
            margin_model=ConstantMarginModel(),
            margin_budget_pct=0.05,
            min_contracts=0,
            max_contracts=None,
        )
    )

    assert decision.contracts == 1
    assert decision.risk_per_contract == pytest.approx(250.0)
    assert decision.risk_scenario == "base"
    assert decision.margin_per_contract == pytest.approx(500.0)
    assert decision.risk_budget_contracts == 4
    assert decision.margin_budget_contracts == 1
    assert decision.sizing_binding_constraint == "margin_budget"
    assert not decision.min_contracts_override_applied


def test_size_entry_intent_reports_min_contracts_override():
    class OneScenarioGenerator:
        def generate(self, *, spec, state):
            _ = (spec, state)
            return (StressScenario(name="base", shock=MarketShock()),)

    class ConstantRiskEstimator:
        def estimate_risk_per_contract(self, *, legs, state, scenarios, pricer):
            _ = (legs, state, scenarios, pricer)
            base = StressScenario(name="base", shock=MarketShock())
            return StressResult(
                worst_loss=2_500.0,
                worst_scenario=base,
                points=(StressPoint(scenario=base, pnl=-2_500.0),),
            )

    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    decision = size_entry_intent(
        SizingRequest(
            intent=_short_straddle_intent(),
            option_contract_multiplier=1,
            spot=100.0,
            volatility=0.2,
            equity=10_000.0,
            pricer=NullPricer(),
            scenario_generator=OneScenarioGenerator(),
            risk_estimator=ConstantRiskEstimator(),
            risk_sizer=RiskBudgetSizer(risk_budget_pct=0.10, min_contracts=1),
            margin_model=None,
            margin_budget_pct=None,
            min_contracts=1,
            max_contracts=None,
        )
    )

    assert decision.contracts == 1
    assert decision.risk_budget_contracts == 0
    assert decision.margin_budget_contracts is None
    assert decision.sizing_binding_constraint == "min_contracts"
    assert decision.min_contracts_override_applied


def test_size_entry_intent_can_use_entry_hedged_risk_basis():
    class SpotShockScenarioGenerator:
        def generate(self, *, spec, state):
            _ = (spec, state)
            return (
                StressScenario(name="down", shock=MarketShock(d_spot=-20.0)),
                StressScenario(name="up", shock=MarketShock(d_spot=20.0)),
            )

    class LinearDeltaRiskEstimator:
        def estimate_risk_per_contract(self, *, legs, state, scenarios, pricer):
            _ = (legs, state, pricer)
            points = tuple(
                StressPoint(
                    scenario=scenario,
                    pnl=50.0 * float(scenario.shock.d_spot),
                )
                for scenario in scenarios
            )
            worst_point = min(points, key=lambda point: point.pnl)
            return StressResult(
                worst_loss=max(-worst_point.pnl, 0.0),
                worst_scenario=worst_point.scenario,
                points=points,
            )

    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    unhedged = size_entry_intent(
        SizingRequest(
            intent=_long_call_intent(),
            option_contract_multiplier=100,
            spot=100.0,
            volatility=0.2,
            equity=10_000.0,
            pricer=NullPricer(),
            scenario_generator=SpotShockScenarioGenerator(),
            risk_estimator=LinearDeltaRiskEstimator(),
            risk_sizer=RiskBudgetSizer(risk_budget_pct=0.10, min_contracts=0),
            margin_model=None,
            margin_budget_pct=None,
            min_contracts=0,
            max_contracts=None,
            entry_risk_basis="unhedged",
        )
    )
    entry_hedged = size_entry_intent(
        SizingRequest(
            intent=_long_call_intent(),
            option_contract_multiplier=100,
            spot=100.0,
            volatility=0.2,
            equity=10_000.0,
            pricer=NullPricer(),
            scenario_generator=SpotShockScenarioGenerator(),
            risk_estimator=LinearDeltaRiskEstimator(),
            risk_sizer=RiskBudgetSizer(risk_budget_pct=0.10, min_contracts=0),
            margin_model=None,
            margin_budget_pct=None,
            min_contracts=0,
            max_contracts=None,
            entry_risk_basis="entry_hedged",
            entry_hedge_target_net_delta=0.0,
            hedge_contract_multiplier=1.0,
        )
    )

    assert unhedged.risk_per_contract == pytest.approx(1_000.0)
    assert entry_hedged.risk_per_contract == pytest.approx(0.0)
    assert unhedged.contracts == 1
    assert entry_hedged.contracts == 0


def test_estimate_entry_intent_margin_per_contract():
    class ConstantMarginModel:
        def initial_margin_requirement(self, *, legs, state, pricer):
            _ = (legs, state, pricer)
            return 456.0

    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    margin_pc = estimate_entry_intent_margin_per_contract(
        intent=_short_straddle_intent(),
        as_of_date=pd.Timestamp("2020-01-01"),
        option_contract_multiplier=1,
        spot=100.0,
        volatility=0.2,
        margin_model=ConstantMarginModel(),
        pricer=NullPricer(),
    )
    assert margin_pc == pytest.approx(456.0)


def test_estimate_entry_intent_margin_uses_each_leg_expiry_when_available():
    class CapturingMarginModel:
        def __init__(self):
            self.time_to_expiry: list[float] = []

        def initial_margin_requirement(self, *, legs, state, pricer):
            _ = (state, pricer)
            self.time_to_expiry = [float(leg.spec.time_to_expiry) for leg in legs]
            return 321.0

    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    margin_model = CapturingMarginModel()
    intent = EntryIntent(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(
            LegSelection(
                spec=LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                quote=_quote(
                    "P",
                    dte=30,
                    expiry_date="2020-01-31",
                    include_yte=False,
                ),
                side=-1,
                entry_price=5.0,
            ),
            LegSelection(
                spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
                quote=_quote(
                    "C",
                    dte=90,
                    expiry_date="2020-03-31",
                    include_yte=False,
                ),
                side=-1,
                entry_price=5.0,
            ),
        ),
    )

    margin_pc = estimate_entry_intent_margin_per_contract(
        intent=intent,
        as_of_date=pd.Timestamp("2020-01-01"),
        option_contract_multiplier=1,
        spot=100.0,
        volatility=0.2,
        margin_model=margin_model,
        pricer=NullPricer(),
    )

    assert margin_pc == pytest.approx(321.0)
    assert len(margin_model.time_to_expiry) == 2
    assert margin_model.time_to_expiry[0] == pytest.approx(30.0 / 365.0, rel=1e-3)
    assert margin_model.time_to_expiry[1] == pytest.approx(90.0 / 365.0, rel=1e-3)
