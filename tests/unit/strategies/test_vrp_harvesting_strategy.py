import pandas as pd
import pytest

from volatility_trading.backtesting import BacktestConfig, MarginPolicy
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.options import (
    MarketShock,
    StressPoint,
    StressResult,
    StressScenario,
)
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingStrategy


def _run_backtest(
    options: pd.DataFrame,
    *,
    rebalance_period: int | None = 5,
    max_holding_period: int | None = None,
    strategy_kwargs: dict | None = None,
):
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        lot_size=1,
        slip_ask=0.0,
        slip_bid=0.0,
        commission_per_leg=0.0,
    )
    strat = VRPHarvestingStrategy(
        signal=ShortOnlySignal(),
        rebalance_period=rebalance_period,
        max_holding_period=max_holding_period,
        **(strategy_kwargs or {}),
    )
    bt = Backtester(
        data={"options": options, "features": None, "hedge": None},
        strategy=strat,
        config=cfg,
    )
    return bt.run()


def _make_options(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df.set_index("trade_date").sort_index()


def test_mtm_delta_pnl_matches_trade_pnl_on_holding_period_exit():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.2,
                "spot_price": 101.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.2,
                "spot_price": 101.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
        ]
    )

    trades, mtm = _run_backtest(options, rebalance_period=2)

    assert len(trades) == 1
    assert trades["pnl"].sum() == pytest.approx(-3.0)
    assert mtm["delta_pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_unresolved_trade_keeps_mtm_path_instead_of_being_dropped():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            # Matching expiry, but not the held strike, so the position never exits.
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 105.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.8,
                "ask_price": 5.0,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 105.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.8,
                "ask_price": 5.0,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 105.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.7,
                "ask_price": 4.9,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 105.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.7,
                "ask_price": 4.9,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
        ]
    )

    trades, mtm = _run_backtest(options, rebalance_period=2)

    assert trades.empty
    assert not mtm.empty
    assert mtm.index.min() == pd.Timestamp("2020-01-01")
    assert mtm.index.max() == pd.Timestamp("2020-01-03")


def test_risk_budget_sizing_sets_contracts_from_worst_loss():
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

    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
        ]
    )

    trades, mtm = _run_backtest(
        options,
        rebalance_period=2,
        strategy_kwargs={
            "risk_budget_pct": 0.10,
            "min_contracts": 0,
            "scenario_generator": OneScenarioGenerator(),
            "risk_estimator": ConstantRiskEstimator(),
        },
    )

    assert len(trades) == 1
    assert trades.iloc[0]["contracts"] == 4
    assert trades.iloc[0]["risk_per_contract"] == pytest.approx(250.0)
    assert trades.iloc[0]["risk_worst_scenario"] == "base"
    assert mtm["delta_pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_margin_budget_caps_contracts_below_risk_budget():
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

    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
        ]
    )

    trades, mtm = _run_backtest(
        options,
        rebalance_period=2,
        strategy_kwargs={
            "risk_budget_pct": 0.10,
            "margin_budget_pct": 0.05,
            "min_contracts": 0,
            "scenario_generator": OneScenarioGenerator(),
            "risk_estimator": ConstantRiskEstimator(),
            "margin_model": ConstantMarginModel(),
        },
    )

    assert len(trades) == 1
    assert trades.iloc[0]["contracts"] == 1
    assert trades.iloc[0]["risk_per_contract"] == pytest.approx(250.0)
    assert trades.iloc[0]["margin_per_contract"] == pytest.approx(500.0)
    assert mtm["delta_pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_margin_call_liquidation_exits_before_holding_period():
    class ConstantMarginModel:
        def initial_margin_requirement(self, *, legs, state, pricer):
            _ = (legs, pricer)
            return 1_000.0 if state.spot <= 105 else 20_000.0

    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 8.0,
                "ask_price": 8.2,
                "spot_price": 118.0,
                "smoothed_iv": 0.32,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 8.0,
                "ask_price": 8.2,
                "spot_price": 118.0,
                "smoothed_iv": 0.32,
            },
        ]
    )

    trades, mtm = _run_backtest(
        options,
        rebalance_period=10,
        strategy_kwargs={
            "margin_model": ConstantMarginModel(),
            "margin_policy": MarginPolicy(margin_call_grace_days=0),
        },
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_type"] == "Margin Call Liquidation"
    assert trades.iloc[0]["contracts"] > 0
    assert mtm.iloc[-1]["open_contracts"] == pytest.approx(0.0)
    assert mtm.iloc[-1]["forced_liquidation"] == pytest.approx(1.0)
    assert mtm["delta_pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_time_to_expiry_prefers_yte_then_dte_then_calendar():
    entry = pd.Timestamp("2020-01-01")
    expiry = pd.Timestamp("2020-01-31")

    yte_first = VRPHarvestingStrategy._time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=0.123,
        quote_dte=30,
    )
    dte_fallback = VRPHarvestingStrategy._time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=float("nan"),
        quote_dte=30,
    )
    calendar_fallback = VRPHarvestingStrategy._time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=0.0,
        quote_dte=None,
    )

    assert yte_first == pytest.approx(0.123)
    assert dte_fallback == pytest.approx(30 / 365.0)
    assert calendar_fallback == pytest.approx(30 / 365.0)


def test_same_day_reentry_can_be_enabled_for_rebalance_rolls():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.0,
                "spot_price": 101.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.0,
                "spot_price": 101.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.4,
                "spot_price": 102.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.4,
                "spot_price": 102.0,
                "smoothed_iv": 0.22,
            },
        ]
    )

    trades_allow, _ = _run_backtest(
        options,
        rebalance_period=1,
        strategy_kwargs={
            "allow_same_day_reentry_on_rebalance": True,
            "allow_same_day_reentry_on_max_holding": False,
        },
    )
    trades_block, _ = _run_backtest(
        options,
        rebalance_period=1,
        strategy_kwargs={
            "allow_same_day_reentry_on_rebalance": False,
            "allow_same_day_reentry_on_max_holding": False,
        },
    )

    assert len(trades_allow) == 2
    assert pd.Timestamp(trades_allow.iloc[1]["entry_date"]) == pd.Timestamp(
        "2020-01-02"
    )
    assert len(trades_block) == 1


def test_rebalance_period_and_max_holding_period_can_be_set_separately():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.0,
                "spot_price": 101.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.0,
                "spot_price": 101.0,
                "smoothed_iv": 0.21,
            },
        ]
    )

    trades, _ = _run_backtest(
        options,
        rebalance_period=1,
        max_holding_period=10,
        strategy_kwargs={
            "allow_same_day_reentry_on_rebalance": False,
            "allow_same_day_reentry_on_max_holding": False,
        },
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_type"] == "Rebalance Period"
