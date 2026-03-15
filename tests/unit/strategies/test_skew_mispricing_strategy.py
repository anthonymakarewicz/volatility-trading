import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    ExecutionConfig,
    MaxHoldingExitRule,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    StopLossExitRule,
    TakeProfitExitRule,
)
from volatility_trading.backtesting.options_engine import (
    BidAskFeeOptionExecutionModel,
    build_options_execution_plan,
)
from volatility_trading.signals import LongOnlySignal, ShortOnlySignal
from volatility_trading.strategies import (
    SkewMispricingSpec,
    make_skew_mispricing_strategy,
)
from volatility_trading.strategies.skew_mispricing.features import (
    build_skew_signal_input,
)


def _make_cfg() -> BacktestRunConfig:
    return BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )


def _make_features(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").sort_index()


def _make_options(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"])
    return frame.set_index("trade_date").sort_index()


def _make_risk_reversal_options() -> pd.DataFrame:
    return _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-02-01",
                "dte": 31,
                "strike": 95.0,
                "option_type": "P",
                "delta": -0.18,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.0,
                "ask_price": 4.2,
                "spot_price": 100.0,
                "market_iv": 0.28,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-02-01",
                "dte": 31,
                "strike": 97.0,
                "option_type": "P",
                "delta": -0.23,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.5,
                "ask_price": 4.7,
                "spot_price": 100.0,
                "market_iv": 0.30,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-02-01",
                "dte": 31,
                "strike": 103.0,
                "option_type": "C",
                "delta": 0.26,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.8,
                "ask_price": 5.0,
                "spot_price": 100.0,
                "market_iv": 0.19,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-02-01",
                "dte": 31,
                "strike": 105.0,
                "option_type": "C",
                "delta": 0.21,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.2,
                "ask_price": 4.4,
                "spot_price": 100.0,
                "market_iv": 0.18,
            },
        ]
    )


def _make_skew_features() -> pd.DataFrame:
    return _make_features(
        [
            {
                "trade_date": "2020-01-01",
                "iv_dlt25_30d": 0.30,
                "iv_dlt75_30d": 0.19,
            }
        ]
    )


def test_build_skew_signal_input_uses_raw_30d_risk_reversal_feature():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-02-01",
                "dte": 31,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.25,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "market_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-02-01",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.25,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.1,
                "ask_price": 5.3,
                "spot_price": 101.0,
                "market_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-02-01",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.25,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.2,
                "ask_price": 5.4,
                "spot_price": 102.0,
                "market_iv": 0.22,
            },
        ]
    )
    features = _make_features(
        [
            {
                "trade_date": "2020-01-01",
                "iv_dlt25_30d": 0.30,
                "iv_dlt75_30d": 0.18,
            },
            {
                "trade_date": "2020-01-02",
                "iv_dlt25_30d": 0.31,
                "iv_dlt75_30d": 0.20,
            },
        ]
    )

    skew = build_skew_signal_input(options, features, None, target_dte=30)

    expected = pd.Series(
        [0.12, 0.11, float("nan")],
        index=pd.Index(
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            name="trade_date",
        ),
        name="skew_rr_30d",
    )
    assert_series_equal(skew, expected)


def test_build_skew_signal_input_requires_features():
    options = _make_risk_reversal_options()

    with pytest.raises(
        ValueError,
        match="requires data.features",
    ):
        build_skew_signal_input(options, None, None, target_dte=30)


def test_build_skew_signal_input_requires_skew_columns():
    options = _make_risk_reversal_options()
    features = _make_features(
        [
            {
                "trade_date": "2020-01-01",
                "iv_dlt25_30d": 0.30,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="iv_dlt75_30d",
    ):
        build_skew_signal_input(options, features, None, target_dte=30)


@pytest.mark.parametrize(
    ("last_value", "expected_column"),
    [
        (2.0, "long"),
        (-1.0, "short"),
    ],
)
def test_skew_default_signal_trades_raw_skew_mean_reversion(
    last_value: float,
    expected_column: str,
) -> None:
    signal = SkewMispricingSpec().signal
    window = int(signal.get_params()["strategy__window"])
    history = ([0.0, 1.0] * ((window + 1) // 2))[:window]
    skew = pd.Series(
        history + [last_value],
        index=pd.date_range("2020-01-01", periods=len(history) + 1, freq="D"),
        name="skew_rr_30d",
    )

    signals = signal.generate_signals(skew)

    assert bool(signals.iloc[-1][expected_column])
    assert not bool(signals.iloc[-1]["exit"])


def test_skew_default_lifecycle_is_signal_driven_with_max_holding_safety_cap():
    lifecycle = SkewMispricingSpec().to_strategy_spec().lifecycle

    assert lifecycle.rebalance_period is None
    assert lifecycle.max_holding_period == 30
    assert tuple(type(rule) for rule in lifecycle.exit_rule_set.rules) == (
        MaxHoldingExitRule,
    )


def test_skew_signal_driven_lifecycle_keeps_max_holding_and_appends_pnl_exits():
    lifecycle = (
        SkewMispricingSpec(
            stop_loss_pnl_per_contract=1.0,
            take_profit_pnl_per_contract=2.0,
            allow_same_day_reentry_on_stop_loss=False,
            allow_same_day_reentry_on_take_profit=True,
        )
        .to_strategy_spec()
        .lifecycle
    )

    assert lifecycle.rebalance_period is None
    assert lifecycle.max_holding_period == 30
    assert tuple(type(rule) for rule in lifecycle.exit_rule_set.rules) == (
        MaxHoldingExitRule,
        StopLossExitRule,
        TakeProfitExitRule,
    )
    assert lifecycle.reentry_policy.allow_on_stop_loss is False
    assert lifecycle.reentry_policy.allow_on_take_profit is True


@pytest.mark.parametrize(
    ("signal", "expected_sides"),
    [
        (LongOnlySignal(), (-1, 1)),
        (ShortOnlySignal(), (1, -1)),
    ],
)
def test_skew_strategy_builds_risk_reversal_via_selector_bands(
    signal,
    expected_sides,
):
    options = _make_risk_reversal_options()
    features = _make_skew_features()
    strategy = make_skew_mispricing_strategy(SkewMispricingSpec(signal=signal))
    plan = build_options_execution_plan(
        spec=strategy,
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(
                chain=options,
                default_contract_multiplier=1.0,
            ),
            features=features,
        ),
        config=_make_cfg(),
        capital=10_000.0,
    )

    setup = plan.hooks.prepare_entry(pd.Timestamp("2020-01-01"), 10_000.0)

    assert setup is not None
    assert len(setup.intent.legs) == 2
    assert [leg.quote.strike for leg in setup.intent.legs] == [97.0, 103.0]
    assert [int(leg.side) for leg in setup.intent.legs] == list(expected_sides)
