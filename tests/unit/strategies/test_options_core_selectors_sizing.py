import pandas as pd
import pytest

from volatility_trading.options import (
    MarketShock,
    OptionType,
    RiskBudgetSizer,
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.strategies.options_core import (
    EntryIntent,
    LegSelection,
    LegSpec,
    estimate_entry_intent_margin_per_contract,
    select_best_expiry_for_leg_group,
    size_entry_intent_contracts,
)


def _quote(option_type: str, *, strike: float = 100.0, dte: int = 30) -> pd.Series:
    return pd.Series(
        {
            "option_type": option_type,
            "strike": strike,
            "dte": dte,
            "yte": dte / 365.0,
            "delta": -0.5 if option_type == "P" else 0.5,
            "bid_price": 5.0,
            "ask_price": 5.2,
        }
    )


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


def test_size_entry_intent_contracts_uses_risk_and_margin_caps():
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

    contracts, risk_pc, risk_scenario, margin_pc = size_entry_intent_contracts(
        intent=_short_straddle_intent(),
        lot_size=1,
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

    assert contracts == 1
    assert risk_pc == pytest.approx(250.0)
    assert risk_scenario == "base"
    assert margin_pc == pytest.approx(500.0)


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
        lot_size=1,
        spot=100.0,
        volatility=0.2,
        margin_model=ConstantMarginModel(),
        pricer=NullPricer(),
    )
    assert margin_pc == pytest.approx(456.0)
