import pandas as pd
import pytest

from volatility_trading.options import (
    MarketShock,
    RiskBudgetSizer,
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.strategies.options_core import (
    choose_expiry_by_target_dte,
    pick_quote_by_delta,
    size_short_straddle_contracts,
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


def test_pick_quote_by_delta_selects_closest_row():
    puts = pd.DataFrame(
        {
            "delta": [-0.35, -0.48, -0.52, -0.70],
            "bid_price": [1.0, 2.0, 3.0, 4.0],
        }
    )
    row = pick_quote_by_delta(puts, target_delta=-0.5, delta_tolerance=0.05)
    assert row is not None
    assert row["delta"] == pytest.approx(-0.48)


def test_choose_expiry_by_target_dte_filters_for_viable_atm_quotes():
    chain = pd.DataFrame(
        {
            "dte": [30, 30, 30, 31, 31, 31],
            "strike": [100.0, 102.0, 98.0, 120.0, 80.0, 130.0],
            "spot_price": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    chosen = choose_expiry_by_target_dte(
        chain,
        target_dte=30,
        max_dte_diff=2,
        min_atm_quotes=2,
    )
    assert chosen == 30


def test_size_short_straddle_contracts_uses_risk_and_margin_caps():
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

    contracts, risk_pc, risk_scenario, margin_pc = size_short_straddle_contracts(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        put_quote=_quote("P"),
        call_quote=_quote("C"),
        put_entry=5.0,
        call_entry=5.0,
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


def test_size_short_straddle_contracts_invalid_market_returns_fallback():
    class NullPricer:
        def price(self, spec, state):
            _ = (spec, state)
            return 1.0

    class EmptyScenarioGenerator:
        def generate(self, *, spec, state):
            _ = (spec, state)
            return ()

    class NeverCalledRiskEstimator:
        def estimate_risk_per_contract(self, *, legs, state, scenarios, pricer):
            _ = (legs, state, scenarios, pricer)
            raise AssertionError("should not be called for invalid market")

    contracts, risk_pc, risk_scenario, margin_pc = size_short_straddle_contracts(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        put_quote=_quote("P"),
        call_quote=_quote("C"),
        put_entry=5.0,
        call_entry=5.0,
        lot_size=1,
        spot=float("nan"),
        volatility=0.2,
        equity=10_000.0,
        pricer=NullPricer(),
        scenario_generator=EmptyScenarioGenerator(),
        risk_estimator=NeverCalledRiskEstimator(),
        risk_sizer=RiskBudgetSizer(risk_budget_pct=0.10, min_contracts=2),
        margin_model=None,
        margin_budget_pct=None,
        min_contracts=0,
        max_contracts=None,
    )

    assert contracts == 2
    assert risk_pc is None
    assert risk_scenario is None
    assert margin_pc is None
