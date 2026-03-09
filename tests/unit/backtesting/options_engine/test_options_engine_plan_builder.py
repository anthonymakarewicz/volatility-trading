import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    BrokerConfig,
    ExecutionConfig,
    HedgeMarketData,
    MarginConfig,
    OptionsBacktestDataBundle,
    OptionsMarketData,
)
from volatility_trading.backtesting.engine import (
    Backtester,
    run_backtest_execution_plan,
)
from volatility_trading.backtesting.options_engine import (
    BidAskFeeOptionExecutionModel,
    ColumnMapOptionsChainAdapter,
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    LegSpec,
    LifecycleConfig,
    OptionExecutionResult,
    OptionsChainAdapterError,
    SizingPolicyConfig,
    StrategySpec,
    StructureSpec,
    build_options_execution_plan,
)
from volatility_trading.options import OptionType
from volatility_trading.signals.base_signal import Signal

# TODO: Consider create a separate unit and integration


class DirectionSignal(Signal):
    def __init__(self, *, direction: int):
        super().__init__()
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        self.direction = direction

    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        if self.direction == 1:
            return pd.DataFrame({"long": True, "short": False}, index=idx)
        return pd.DataFrame({"long": False, "short": True}, index=idx)

    def get_params(self) -> dict:
        return {"direction": self.direction}

    def set_params(self, **kwargs):
        if "direction" in kwargs:
            self.direction = int(kwargs["direction"])


def _make_options() -> pd.DataFrame:
    rows = [
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
            "market_iv": 0.20,
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
            "bid_price": 6.0,
            "ask_price": 6.2,
            "spot_price": 101.0,
            "market_iv": 0.21,
        },
    ]
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df.set_index("trade_date").sort_index()


def _run_strategy(direction: int):
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="directional_test",
        signal=DirectionSignal(direction=direction),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(chain=_make_options())
        ),
        strategy=spec,
        config=cfg,
    )
    return bt.run()


def _make_hedge_market(options: pd.DataFrame) -> HedgeMarketData:
    dates = options.index.unique()
    mid = pd.Series(100.0, index=dates)
    return HedgeMarketData(mid=mid)


def _make_options_alias_columns() -> pd.DataFrame:
    rows = [
        {
            "date": "2020-01-01",
            "expiry": "2020-01-31",
            "dte": 30,
            "strike": 100.0,
            "option_type": "C",
            "delta": 0.5,
            "bid": 5.0,
            "ask": 5.2,
        },
        {
            "date": "2020-01-02",
            "expiry": "2020-01-31",
            "dte": 29,
            "strike": 100.0,
            "option_type": "C",
            "delta": 0.5,
            "bid": 6.0,
            "ask": 6.2,
        },
    ]
    return pd.DataFrame(rows)


def _make_options_column_map_only_columns() -> pd.DataFrame:
    rows = [
        {
            "qdt": "2020-01-01",
            "exp": "2020-01-31",
            "days": 30,
            "cp": "C",
            "k": 100.0,
            "d": 0.5,
            "b": 5.0,
            "a": 5.2,
        },
        {
            "qdt": "2020-01-02",
            "exp": "2020-01-31",
            "days": 29,
            "cp": "C",
            "k": 100.0,
            "d": 0.5,
            "b": 6.0,
            "a": 6.2,
        },
    ]
    return pd.DataFrame(rows)


def test_config_strategy_uses_long_direction_for_entry_side():
    trades, _ = _run_strategy(direction=1)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert leg["side"] == 1
    assert leg["effective_side"] == 1
    assert row["pnl"] == pytest.approx(0.8)


def test_config_strategy_uses_short_direction_for_entry_side():
    trades, _ = _run_strategy(direction=-1)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert leg["side"] == -1
    assert leg["effective_side"] == -1
    assert row["pnl"] == pytest.approx(-1.2)


@pytest.mark.parametrize(
    (
        "direction",
        "expected_trade_pnl",
        "expected_side",
        "expected_effective_side",
        "expected_entry_price",
        "expected_exit_price",
        "expected_delta",
        "expected_gamma",
        "expected_vega",
        "expected_theta",
        "expected_equity_day2",
    ),
    [
        (
            1,
            0.8,
            1,
            1,
            5.2,
            6.0,
            0.5,
            0.01,
            0.10,
            -0.02,
            10_000.7,
        ),
        (
            -1,
            -1.2,
            -1,
            -1,
            5.0,
            6.2,
            -0.5,
            -0.01,
            -0.10,
            0.02,
            9_998.7,
        ),
    ],
)
def test_directional_strategy_outputs_regression_snapshot(
    direction: int,
    expected_trade_pnl: float,
    expected_side: int,
    expected_effective_side: int,
    expected_entry_price: float,
    expected_exit_price: float,
    expected_delta: float,
    expected_gamma: float,
    expected_vega: float,
    expected_theta: float,
    expected_equity_day2: float,
):
    trades, mtm = _run_strategy(direction=direction)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert row["entry_date"] == pd.Timestamp("2020-01-01")
    assert row["exit_date"] == pd.Timestamp("2020-01-02")
    assert row["exit_type"] == "Rebalance Period"
    assert row["pnl"] == pytest.approx(expected_trade_pnl)
    assert leg["side"] == expected_side
    assert leg["effective_side"] == expected_effective_side
    assert leg["entry_price"] == pytest.approx(expected_entry_price)
    assert leg["exit_price"] == pytest.approx(expected_exit_price)
    assert leg["entry_mid_price"] == pytest.approx(5.1)
    assert leg["exit_mid_price"] == pytest.approx(6.1)
    assert row["option_entry_cost"] == pytest.approx(0.1)
    assert row["option_exit_cost"] == pytest.approx(0.1)

    expected_mtm = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "delta_pnl": -0.1,
                "option_market_pnl": 0.0,
                "option_trade_cost": 0.1,
                "delta": expected_delta,
                "net_delta": expected_delta,
                "gamma": expected_gamma,
                "vega": expected_vega,
                "theta": expected_theta,
                "S": 100.0,
                "iv": 0.20,
                "open_contracts": 1,
                "equity": 9_999.9,
            },
            {
                "date": pd.Timestamp("2020-01-02"),
                "delta_pnl": expected_trade_pnl,
                "option_market_pnl": expected_trade_pnl + 0.2,
                "option_trade_cost": 0.2,
                "delta": expected_delta,
                "net_delta": expected_delta,
                "gamma": expected_gamma,
                "vega": expected_vega,
                "theta": expected_theta,
                "S": 101.0,
                "iv": 0.21,
                "open_contracts": 1,
                "equity": expected_equity_day2,
            },
        ]
    ).set_index("date")

    mtm_subset = mtm[
        [
            "delta_pnl",
            "option_market_pnl",
            "option_trade_cost",
            "delta",
            "net_delta",
            "gamma",
            "vega",
            "theta",
            "S",
            "iv",
            "open_contracts",
            "equity",
        ]
    ]
    assert_frame_equal(
        mtm_subset,
        expected_mtm,
        check_dtype=False,
        atol=1e-12,
        rtol=0.0,
    )


def test_margin_budget_requires_broker_margin_model():
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="margin_budget_requires_model",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
        sizing=SizingPolicyConfig(margin_budget_pct=0.10),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
        broker=BrokerConfig(margin=MarginConfig(model=None)),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(chain=_make_options())
        ),
        strategy=spec,
        config=cfg,
    )
    with pytest.raises(
        ValueError,
        match="strategy margin_budget_pct requires config.broker.margin.model",
    ):
        bt.run()


def test_enabled_delta_hedging_requires_hedge_market_data():
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="delta_hedge_requires_data",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(
            rebalance_period=1,
            delta_hedge=DeltaHedgePolicy(
                enabled=True,
                trigger=HedgeTriggerPolicy(
                    band_model=FixedDeltaBandModel(half_width_abs=0.1)
                ),
            ),
        ),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(chain=_make_options())
        ),
        strategy=spec,
        config=cfg,
    )
    with pytest.raises(
        ValueError,
        match="enabled delta hedging requires data.hedge_market",
    ):
        bt.run()


def test_enabled_delta_hedging_accepts_complete_hedge_market_data():
    options = _make_options()
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="delta_hedge_with_data",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(
            rebalance_period=1,
            delta_hedge=DeltaHedgePolicy(
                enabled=True,
                trigger=HedgeTriggerPolicy(
                    band_model=FixedDeltaBandModel(half_width_abs=0.1)
                ),
            ),
        ),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(chain=options),
            hedge_market=_make_hedge_market(options),
        ),
        strategy=spec,
        config=cfg,
    )
    trades, mtm = bt.run()

    assert len(trades) == 1
    assert len(mtm) == 2


def test_default_orats_adapter_raises_on_non_orats_alias_dataset():
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="default_orats_non_orats_alias",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(
                chain=_make_options_column_map_only_columns()
            )
        ),
        strategy=spec,
        config=cfg,
    )
    with pytest.raises(
        OptionsChainAdapterError,
        match="missing required canonical columns",
    ):
        bt.run()


def test_runtime_data_options_adapter_is_used_when_provided():
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="runtime_adapter_from_options_market_data",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(
                chain=_make_options_column_map_only_columns(),
                options_adapter=ColumnMapOptionsChainAdapter(
                    source_to_canonical={
                        "qdt": "trade_date",
                        "exp": "expiry_date",
                        "days": "dte",
                        "cp": "option_type",
                        "k": "strike",
                        "d": "delta",
                        "b": "bid_price",
                        "a": "ask_price",
                    }
                ),
            )
        ),
        strategy=spec,
        config=cfg,
    )
    trades, mtm = bt.run()

    assert len(trades) == 1
    assert len(mtm) == 2


def test_runtime_adapter_from_options_market_data_is_used_when_provided():
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="runtime_adapter_from_options_market_data",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
    )
    adapter = ColumnMapOptionsChainAdapter(
        source_to_canonical={
            "date": "trade_date",
            "expiry": "expiry_date",
            "dte": "dte",
            "option_type": "option_type",
            "strike": "strike",
            "delta": "delta",
            "bid": "bid_price",
            "ask": "ask_price",
        }
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=0.0,
            ),
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(
                chain=_make_options_alias_columns(),
                symbol="SPY",
                default_contract_multiplier=100.0,
                options_adapter=adapter,
            ),
        ),
        strategy=spec,
        config=cfg,
    )
    trades, mtm = bt.run()

    assert len(trades) == 1
    assert len(mtm) == 2


def test_build_plan_supports_custom_option_execution_model_injection():
    class _DirectionalCostOptionExecutionModel:
        def execute(self, *, order):
            if order.trade_side > 0:
                return OptionExecutionResult(
                    fill_price=10.0,
                    total_cost=2.0,
                    price_cost=2.0,
                    fee_cost=0.0,
                )
            return OptionExecutionResult(
                fill_price=1.0,
                total_cost=3.0,
                price_cost=3.0,
                fee_cost=0.0,
            )

    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="custom_option_exec_injection",
        signal=DirectionSignal(direction=1),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1, max_holding_period=None),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            option_execution_model=_DirectionalCostOptionExecutionModel(),
        ),
    )
    plan = build_options_execution_plan(
        spec=spec,
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(chain=_make_options())
        ),
        config=cfg,
        capital=10_000.0,
    )
    trades, mtm = run_backtest_execution_plan(plan)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert leg["entry_price"] == pytest.approx(10.0)
    assert leg["exit_price"] == pytest.approx(1.0)
    assert row["option_entry_cost"] == pytest.approx(2.0)
    assert row["option_exit_cost"] == pytest.approx(3.0)
    assert row["pnl"] == pytest.approx(-4.0)
    assert mtm.loc[pd.Timestamp("2020-01-01"), "option_trade_cost"] == pytest.approx(
        2.0
    )
