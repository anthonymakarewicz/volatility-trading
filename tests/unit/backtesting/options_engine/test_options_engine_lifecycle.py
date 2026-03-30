import pandas as pd
import pytest

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    ExecutionConfig,
    HedgeMarketData,
    MarginPolicy,
)
from volatility_trading.backtesting.options_engine import (
    BidAskFeeOptionExecutionModel,
    DeltaHedgePolicy,
    ExitRuleSet,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    HedgeExecutionModel,
    HedgeExecutionResult,
    HedgeTriggerPolicy,
    LegSpec,
    MidNoCostHedgeExecutionModel,
    OptionExecutionModel,
    OptionExecutionResult,
    StopLossExitRule,
    TakeProfitExitRule,
    WWDeltaBandModel,
)
from volatility_trading.backtesting.options_engine.contracts.market import QuoteSnapshot
from volatility_trading.backtesting.options_engine.contracts.runtime import (
    PositionEntrySetup,
)
from volatility_trading.backtesting.options_engine.contracts.structures import (
    EntryIntent,
    LegSelection,
)
from volatility_trading.backtesting.options_engine.lifecycle.engine import (
    PositionLifecycleEngine,
)
from volatility_trading.options import MarketState, OptionType


class _NullPricer:
    def price(self, spec, state):
        _ = (spec, state)
        return 1.0


class _ConstantMarginModel:
    def __init__(self, margin_per_contract: float):
        self.margin_per_contract = float(margin_per_contract)

    def initial_margin_requirement(self, *, legs, state, pricer):
        _ = (legs, state, pricer)
        return self.margin_per_contract


def _make_cfg(
    *,
    commission_per_leg: float = 0.0,
    hedge_slip_ask: float = 0.0,
    hedge_slip_bid: float = 0.0,
    hedge_fee_bps: float = 0.0,
) -> BacktestRunConfig:
    return BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            option_execution_model=BidAskFeeOptionExecutionModel(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_leg=commission_per_leg,
            ),
            hedge_execution_model=FixedBpsHedgeExecutionModel(
                slip_ask=hedge_slip_ask,
                slip_bid=hedge_slip_bid,
                fee_bps=hedge_fee_bps,
            ),
        ),
    )


def _make_quote(
    *,
    strike: float = 100.0,
    option_type: str = "C",
    bid_price: float = 5.0,
    ask_price: float = 5.0,
    expiry_date: str = "2020-01-31",
    dte: int = 30,
    delta: float = 0.5,
    gamma: float = 0.1,
    vega: float = 0.2,
    theta: float = -0.3,
    spot_price: float = 100.0,
    market_iv: float = 0.2,
) -> QuoteSnapshot:
    return QuoteSnapshot.from_series(
        pd.Series(
            {
                "strike": strike,
                "option_type": option_type,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "expiry_date": pd.Timestamp(expiry_date),
                "dte": dte,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "spot_price": spot_price,
                "market_iv": market_iv,
            }
        )
    )


def _make_setup(
    *,
    contracts: int = 1,
    entry_price: float = 5.0,
    margin_per_contract: float | None = None,
) -> PositionEntrySetup:
    quote = _make_quote()
    intent = EntryIntent(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(
            LegSelection(
                spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
                quote=quote,
                side=-1,
                entry_price=entry_price,
            ),
        ),
        entry_state=MarketState(spot=100.0, volatility=0.2),
    )
    return PositionEntrySetup(
        intent=intent,
        contracts=contracts,
        risk_per_contract=None,
        risk_worst_scenario=None,
        margin_per_contract=margin_per_contract,
    )


def _make_two_leg_setup(*, contracts: int = 1) -> PositionEntrySetup:
    put_quote = _make_quote(option_type="P", delta=-0.5)
    call_quote = _make_quote(option_type="C", delta=0.5)
    intent = EntryIntent(
        entry_date=pd.Timestamp("2020-01-01"),
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(
            LegSelection(
                spec=LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                quote=put_quote,
                side=-1,
                entry_price=5.0,
            ),
            LegSelection(
                spec=LegSpec(option_type=OptionType.CALL, delta_target=0.5),
                quote=call_quote,
                side=-1,
                entry_price=5.0,
            ),
        ),
        entry_state=MarketState(spot=100.0, volatility=0.2),
    )
    return PositionEntrySetup(
        intent=intent,
        contracts=contracts,
        risk_per_contract=None,
        risk_worst_scenario=None,
        margin_per_contract=None,
    )


def _make_options_row_for_date(
    *,
    trade_date: str = "2020-01-02",
    strike: float = 100.0,
    option_type: str = "C",
    bid_price: float = 5.0,
    ask_price: float = 5.0,
    spot_price: float = 101.0,
    market_iv: float = 0.21,
) -> pd.DataFrame:
    row = {
        "trade_date": trade_date,
        "expiry_date": "2020-01-31",
        "dte": 29,
        "strike": strike,
        "option_type": option_type,
        "delta": 0.5,
        "gamma": 0.1,
        "vega": 0.2,
        "theta": -0.3,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "spot_price": spot_price,
        "market_iv": market_iv,
    }
    options = pd.DataFrame([row])
    options["trade_date"] = pd.to_datetime(options["trade_date"])
    options["expiry_date"] = pd.to_datetime(options["expiry_date"])
    return options.set_index("trade_date").sort_index()


def _make_engine(
    *,
    rebalance_period: int | None = 5,
    option_contract_multiplier: float = 1.0,
    exit_rule_set: ExitRuleSet | None = None,
    margin_policy: MarginPolicy | None = None,
    margin_model=None,
    delta_hedge_policy: DeltaHedgePolicy | None = None,
    hedge_market: HedgeMarketData | None = None,
    hedge_execution_model: HedgeExecutionModel | None = None,
    option_execution_model: OptionExecutionModel | None = None,
) -> PositionLifecycleEngine:
    lifecycle_kwargs = {}
    if hedge_execution_model is not None:
        lifecycle_kwargs["hedge_execution_model"] = hedge_execution_model
    if option_execution_model is not None:
        lifecycle_kwargs["option_execution_model"] = option_execution_model
    return PositionLifecycleEngine(
        rebalance_period=rebalance_period,
        max_holding_period=None,
        exit_rule_set=exit_rule_set or ExitRuleSet.period_rules(),
        margin_policy=margin_policy,
        margin_model=margin_model,
        pricer=_NullPricer(),
        delta_hedge_policy=delta_hedge_policy or DeltaHedgePolicy(),
        option_contract_multiplier=option_contract_multiplier,
        hedge_market=hedge_market,
        **lifecycle_kwargs,
    )


def _make_hedge_market(
    prices: dict[str, float],
    *,
    bid_prices: dict[str, float] | None = None,
    ask_prices: dict[str, float] | None = None,
    contract_multiplier: float = 1.0,
) -> HedgeMarketData:
    def _to_series(values: dict[str, float] | None) -> pd.Series | None:
        if values is None:
            return None
        series = pd.Series(values, dtype=float)
        series.index = pd.to_datetime(series.index)
        return series.sort_index()

    mid = _to_series(prices)
    assert mid is not None  # nosec B101 - helper contract requires prices
    return HedgeMarketData(
        mid=mid,
        bid=_to_series(bid_prices),
        ask=_to_series(ask_prices),
        contract_multiplier=contract_multiplier,
    )


def test_open_position_records_entry_commission_and_greeks():
    setup = _make_setup(contracts=3)
    cfg = _make_cfg(commission_per_leg=1.0)
    engine = _make_engine()

    position, entry_record = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert position.contracts_open == 3
    assert position.prev_mtm == pytest.approx(0.0)
    assert entry_record.open_contracts == 3
    assert entry_record.option_trade_cost == pytest.approx(3.0)
    assert entry_record.option_market_pnl == pytest.approx(0.0)
    assert entry_record.delta_pnl == pytest.approx(-3.0)
    assert entry_record.greeks.delta == pytest.approx(-1.5)
    assert entry_record.net_delta == pytest.approx(-1.5)
    assert entry_record.greeks.gamma == pytest.approx(-0.3)
    assert entry_record.greeks.vega == pytest.approx(-0.6)
    assert entry_record.greeks.theta == pytest.approx(0.9)


def test_open_position_commission_scales_with_leg_count():
    setup = _make_two_leg_setup(contracts=2)
    cfg = _make_cfg(commission_per_leg=1.0)
    engine = _make_engine()

    _, entry_record = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    # Entry booking only: 2 legs * $1 commission * 2 contracts.
    assert entry_record.option_trade_cost == pytest.approx(4.0)
    assert entry_record.delta_pnl == pytest.approx(-4.0)


def test_option_contract_multiplier_scales_option_pnl_and_trade_pnl():
    setup = _make_setup(contracts=2)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=1,
        option_contract_multiplier=50.0,
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            bid_price=6.0,
            ask_price=6.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    assert step_result.trade_rows[0].pnl == pytest.approx(-100.0)
    assert step_result.mtm_record.option_market_pnl == pytest.approx(-100.0)
    assert step_result.mtm_record.delta_pnl == pytest.approx(-100.0)


def test_mark_position_with_missing_quotes_keeps_position_open():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(rebalance_period=1)
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date(
        strike=105.0,
        bid_price=4.0,
        ask_price=4.2,
    )
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
    )
    updated_position = step_result.position
    mtm_record = step_result.mtm_record
    trade_rows = step_result.trade_rows

    assert updated_position is not None
    assert trade_rows == []
    assert mtm_record.delta_pnl == pytest.approx(0.0)
    assert mtm_record.open_contracts == 1
    assert updated_position.prev_mtm == pytest.approx(0.0)
    assert updated_position.last_greeks.delta == pytest.approx(-0.5)


def test_mark_position_missing_quotes_after_expiry_settles_position_intrinsically():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(rebalance_period=None)
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date(
        trade_date="2020-02-03",
        strike=105.0,
        bid_price=4.0,
        ask_price=4.2,
        spot_price=101.0,
    )
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-02-03"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    assert step_result.trade_rows[0].exit_type == "Expiry Settlement"
    assert step_result.trade_rows[0].pnl == pytest.approx(4.0)
    assert step_result.trade_rows[0].trade_legs[0].exit_price == pytest.approx(1.0)
    assert step_result.mtm_record.open_contracts == 0
    assert step_result.mtm_record.delta_pnl == pytest.approx(4.0)
    assert step_result.mtm_record.net_delta == pytest.approx(0.0)


def test_mark_position_rebalance_exit_closes_position_and_emits_trade():
    setup = _make_setup(contracts=2)
    cfg = _make_cfg()
    engine = _make_engine(rebalance_period=1)
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date(
        bid_price=6.0,
        ask_price=6.0,
    )
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
    )
    updated_position = step_result.position
    mtm_record = step_result.mtm_record
    trade_rows = step_result.trade_rows

    assert updated_position is None
    assert len(trade_rows) == 1
    assert trade_rows[0].exit_type == "Rebalance Period"
    assert trade_rows[0].contracts == 2
    assert trade_rows[0].pnl == pytest.approx(-2.0)
    assert mtm_record.delta_pnl == pytest.approx(-2.0)
    assert mtm_record.open_contracts == 0
    assert mtm_record.greeks.delta == pytest.approx(0.0)
    assert mtm_record.net_delta == pytest.approx(0.0)


def test_mark_position_signal_exit_override_takes_precedence_over_periodic_exit():
    setup = _make_setup(contracts=2)
    cfg = _make_cfg()
    engine = _make_engine(rebalance_period=1)
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date(
        bid_price=6.0,
        ask_price=6.0,
    )
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
        exit_type_override="Signal Exit",
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    assert step_result.trade_rows[0].exit_type == "Signal Exit"


def test_mark_position_stop_loss_exit_uses_pnl_per_contract():
    setup = _make_setup(contracts=2)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        exit_rule_set=ExitRuleSet(
            rules=(StopLossExitRule(threshold_per_contract=0.5),)
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            bid_price=6.0,
            ask_price=6.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    assert step_result.trade_rows[0].exit_type == "Stop Loss"
    assert step_result.trade_rows[0].pnl == pytest.approx(-2.0)


def test_mark_position_take_profit_exit_uses_pnl_per_contract():
    setup = _make_setup(contracts=2)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        exit_rule_set=ExitRuleSet(
            rules=(TakeProfitExitRule(threshold_per_contract=0.5),)
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            bid_price=4.0,
            ask_price=4.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    assert step_result.trade_rows[0].exit_type == "Take Profit"
    assert step_result.trade_rows[0].pnl == pytest.approx(2.0)


def test_mark_position_forced_liquidation_full_mode_closes_all_contracts():
    setup = _make_setup(contracts=3, margin_per_contract=1_000.0)
    cfg = _make_cfg()
    policy = MarginPolicy(maintenance_margin_ratio=1.0, margin_call_grace_days=0)
    margin_model = _ConstantMarginModel(margin_per_contract=10_000.0)
    engine = _make_engine(
        rebalance_period=10,
        margin_policy=policy,
        margin_model=margin_model,
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date()
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
    )
    updated_position = step_result.position
    mtm_record = step_result.mtm_record
    trade_rows = step_result.trade_rows

    assert updated_position is None
    assert len(trade_rows) == 1
    assert trade_rows[0].exit_type == "Margin Call Liquidation"
    assert trade_rows[0].contracts == 3
    assert mtm_record.open_contracts == 0
    assert mtm_record.margin.core.forced_liquidation is True
    assert mtm_record.margin.core.contracts_liquidated == 3


def test_mark_position_forced_liquidation_target_mode_can_be_partial():
    setup = _make_setup(contracts=4, margin_per_contract=1_000.0)
    cfg = _make_cfg()
    policy = MarginPolicy(
        maintenance_margin_ratio=1.0,
        margin_call_grace_days=0,
        liquidation_mode="target",
    )
    margin_model = _ConstantMarginModel(margin_per_contract=5_000.0)
    engine = _make_engine(
        rebalance_period=10,
        margin_policy=policy,
        margin_model=margin_model,
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=12_000.0,
    )

    options = _make_options_row_for_date()
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=12_000.0,
    )
    updated_position = step_result.position
    mtm_record = step_result.mtm_record
    trade_rows = step_result.trade_rows

    assert updated_position is not None
    assert len(trade_rows) == 1
    assert trade_rows[0].exit_type == "Margin Call Partial Liquidation"
    assert trade_rows[0].contracts == 2
    assert updated_position.contracts_open == 2
    assert mtm_record.open_contracts == 2
    assert mtm_record.margin.core.contracts_liquidated == 2
    assert mtm_record.margin.core.forced_liquidation is True


def test_mark_position_applies_delta_band_hedging_and_updates_net_delta():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 101.0,
            }
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    options = _make_options_row_for_date(
        trade_date="2020-01-02",
        spot_price=101.0,
    )
    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=options,
        cfg=cfg,
        equity_running=10_000.0,
    )
    updated_position = step_result.position
    mtm_record = step_result.mtm_record

    assert updated_position is not None
    assert mtm_record.hedge_qty == pytest.approx(0.5)
    assert mtm_record.net_delta == pytest.approx(0.0)
    assert mtm_record.hedge_pnl == pytest.approx(0.0)
    assert mtm_record.hedge_carry_pnl == pytest.approx(0.0)
    assert mtm_record.hedge_trade_cost == pytest.approx(0.0)
    assert mtm_record.hedge_turnover == pytest.approx(0.5)
    assert mtm_record.hedge_trade_count == 1
    assert updated_position.hedge.qty == pytest.approx(0.5)
    assert updated_position.last_net_delta == pytest.approx(0.0)
    assert updated_position.hedge.last_price == pytest.approx(101.0)
    assert updated_position.hedge.last_rebalance_date == pd.Timestamp("2020-01-02")


def test_mark_position_applies_time_based_hedging_trigger():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=None,
                rebalance_every_n_days=2,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 101.0,
                "2020-01-03": 102.0,
            }
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=101.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )
    position_day2 = day2.position
    assert position_day2 is not None
    assert day2.mtm_record.hedge_qty == pytest.approx(0.0)
    assert day2.mtm_record.net_delta == pytest.approx(-0.5)
    assert day2.mtm_record.hedge_trade_count == 0
    assert day2.mtm_record.hedge_turnover == pytest.approx(0.0)

    day3 = engine.mark_position(
        position=position_day2,
        curr_date=pd.Timestamp("2020-01-03"),
        options=_make_options_row_for_date(
            trade_date="2020-01-03",
            spot_price=102.0,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(day2.mtm_record.delta_pnl),
    )
    position_day3 = day3.position
    assert position_day3 is not None
    assert day3.mtm_record.hedge_qty == pytest.approx(0.5)
    assert day3.mtm_record.net_delta == pytest.approx(0.0)
    assert day3.mtm_record.hedge_trade_count == 1
    assert day3.mtm_record.hedge_turnover == pytest.approx(0.5)
    assert position_day3.hedge.last_rebalance_date == pd.Timestamp("2020-01-03")


def test_mark_position_time_trigger_recenters_with_fixed_band_center_mode():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=1.0),
                rebalance_every_n_days=1,
                combine_mode="or",
            ),
            rebalance_to="center",
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 101.0,
            }
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=101.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.mtm_record.hedge_qty == pytest.approx(0.5)
    assert step_result.mtm_record.net_delta == pytest.approx(0.0)
    assert step_result.mtm_record.hedge_trade_count == 1


def test_mark_position_records_hedge_carry_pnl_without_rebalance_trade():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 101.0,
                "2020-01-03": 102.0,
            }
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=101.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )
    position_day2 = day2.position
    assert position_day2 is not None

    day3 = engine.mark_position(
        position=position_day2,
        curr_date=pd.Timestamp("2020-01-03"),
        options=_make_options_row_for_date(
            trade_date="2020-01-03",
            spot_price=102.0,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(day2.mtm_record.delta_pnl),
    )

    assert day3.mtm_record.hedge_qty == pytest.approx(0.5)
    assert day3.mtm_record.hedge_carry_pnl == pytest.approx(0.5)
    assert day3.mtm_record.hedge_trade_cost == pytest.approx(0.0)
    assert day3.mtm_record.hedge_turnover == pytest.approx(0.0)
    assert day3.mtm_record.hedge_trade_count == 0
    assert day3.mtm_record.hedge_pnl == pytest.approx(0.5)


def test_standard_exit_trade_pnl_includes_cumulative_hedge_pnl():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 100.0,
                "2020-01-03": 100.0,
            },
            bid_prices={
                "2020-01-02": 99.0,
                "2020-01-03": 100.0,
            },
            ask_prices={
                "2020-01-02": 101.0,
                "2020-01-03": 100.0,
            },
        ),
    )
    position, entry_record = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            bid_price=5.0,
            ask_price=5.0,
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )
    assert day2.position is not None
    assert day2.mtm_record.hedge_trade_cost == pytest.approx(0.5)

    day3 = engine.mark_position(
        position=day2.position,
        curr_date=pd.Timestamp("2020-01-03"),
        options=_make_options_row_for_date(
            trade_date="2020-01-03",
            bid_price=5.0,
            ask_price=5.0,
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(day2.mtm_record.delta_pnl),
        exit_type_override="Signal Exit",
    )

    assert day3.position is None
    assert len(day3.trade_rows) == 1
    assert day3.trade_rows[0].pnl == pytest.approx(-0.5)
    total_realized = (
        float(entry_record.delta_pnl)
        + float(day2.mtm_record.delta_pnl)
        + float(day3.mtm_record.delta_pnl)
    )
    assert total_realized == pytest.approx(day3.trade_rows[0].pnl)


def test_stop_loss_exit_uses_cumulative_hedge_pnl():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        exit_rule_set=ExitRuleSet(
            rules=(StopLossExitRule(threshold_per_contract=0.75),)
        ),
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 100.0,
                "2020-01-03": 99.4,
            },
            bid_prices={
                "2020-01-02": 99.0,
                "2020-01-03": 99.4,
            },
            ask_prices={
                "2020-01-02": 101.0,
                "2020-01-03": 99.4,
            },
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            bid_price=5.0,
            ask_price=5.0,
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )
    assert day2.position is not None

    day3 = engine.mark_position(
        position=day2.position,
        curr_date=pd.Timestamp("2020-01-03"),
        options=_make_options_row_for_date(
            trade_date="2020-01-03",
            bid_price=5.0,
            ask_price=5.0,
            spot_price=99.4,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(day2.mtm_record.delta_pnl),
    )

    assert day3.position is None
    assert len(day3.trade_rows) == 1
    assert day3.trade_rows[0].exit_type == "Stop Loss"
    assert day3.trade_rows[0].pnl == pytest.approx(-0.8)


def test_mark_position_scales_hedge_qty_and_pnl_by_contract_multiplier():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {
                "2020-01-02": 101.0,
                "2020-01-03": 102.0,
            },
            contract_multiplier=50.0,
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=101.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )
    assert day2.mtm_record.hedge_qty == pytest.approx(0.01)
    assert day2.mtm_record.net_delta == pytest.approx(0.0)
    assert day2.mtm_record.hedge_turnover == pytest.approx(0.01)
    assert day2.position is not None

    day3 = engine.mark_position(
        position=day2.position,
        curr_date=pd.Timestamp("2020-01-03"),
        options=_make_options_row_for_date(
            trade_date="2020-01-03",
            spot_price=102.0,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(day2.mtm_record.delta_pnl),
    )

    assert day3.mtm_record.hedge_carry_pnl == pytest.approx(0.5)
    assert day3.mtm_record.hedge_trade_cost == pytest.approx(0.0)
    assert day3.mtm_record.hedge_pnl == pytest.approx(0.5)


def test_standard_exit_trade_pnl_includes_cumulative_financing_pnl():
    setup = _make_setup(contracts=1, margin_per_contract=20_000.0)
    cfg = _make_cfg()
    policy = MarginPolicy(
        apply_financing=True,
        cash_rate_annual=0.0,
        borrow_rate_annual=0.252,
        trading_days_per_year=252,
        maintenance_margin_ratio=0.0,
    )
    engine = _make_engine(
        rebalance_period=10,
        margin_policy=policy,
    )
    position, entry_record = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    day2 = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            bid_price=5.0,
            ask_price=5.0,
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0 + float(entry_record.delta_pnl),
        exit_type_override="Signal Exit",
    )

    assert day2.position is None
    assert len(day2.trade_rows) == 1
    assert entry_record.delta_pnl == pytest.approx(-10.0)
    assert day2.mtm_record.margin.core.financing_pnl == pytest.approx(-10.01)
    assert day2.trade_rows[0].pnl == pytest.approx(-20.01)
    total_realized = float(entry_record.delta_pnl) + float(day2.mtm_record.delta_pnl)
    assert total_realized == pytest.approx(day2.trade_rows[0].pnl)


def test_mark_position_uses_bid_ask_hedge_cost_when_quotes_available():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {"2020-01-02": 100.0},
            bid_prices={"2020-01-02": 99.0},
            ask_prices={"2020-01-02": 101.0},
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.mtm_record.hedge_trade_cost == pytest.approx(0.5)
    assert step_result.mtm_record.hedge_pnl == pytest.approx(-0.5)
    assert step_result.mtm_record.net_delta == pytest.approx(0.0)


def test_mark_position_applies_fixed_bps_fee_on_hedge_notional():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg(hedge_fee_bps=10.0)
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {"2020-01-02": 100.0},
            bid_prices={"2020-01-02": 99.0},
            ask_prices={"2020-01-02": 101.0},
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    # Trade qty=0.5, fill=101, mid=100 -> spread/slippage cost = 0.5.
    # Fee = 10 bps * (0.5 * 101) = 0.0505. Total = 0.5505.
    assert step_result.mtm_record.hedge_trade_cost == pytest.approx(0.5505)
    assert step_result.mtm_record.hedge_pnl == pytest.approx(-0.5505)


def test_mark_position_supports_mid_no_cost_execution_model():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg(hedge_fee_bps=25.0, hedge_slip_ask=0.25, hedge_slip_bid=0.25)
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market(
            {"2020-01-02": 100.0},
            bid_prices={"2020-01-02": 99.0},
            ask_prices={"2020-01-02": 101.0},
        ),
        hedge_execution_model=MidNoCostHedgeExecutionModel(),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=100.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.mtm_record.hedge_trade_cost == pytest.approx(0.0)
    assert step_result.mtm_record.hedge_pnl == pytest.approx(0.0)
    assert step_result.mtm_record.net_delta == pytest.approx(0.0)


def test_mark_position_supports_custom_hedge_execution_model():
    class _FixedCostExecutionModel:
        def execute(
            self,
            *,
            trade_qty: float,
            hedge_market,
        ) -> HedgeExecutionResult:
            _ = (hedge_market,)
            return HedgeExecutionResult(
                fill_price=101.0,
                total_cost=abs(trade_qty) * 0.1,
            )

    setup = _make_setup(contracts=1)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=0.1),
                rebalance_every_n_days=None,
            ),
        ),
        hedge_market=_make_hedge_market({"2020-01-02": 101.0}),
        hedge_execution_model=_FixedCostExecutionModel(),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=101.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    updated_position = step_result.position
    assert updated_position is not None
    assert step_result.mtm_record.hedge_qty == pytest.approx(0.5)
    assert step_result.mtm_record.net_delta == pytest.approx(0.0)
    assert step_result.mtm_record.hedge_carry_pnl == pytest.approx(0.0)
    assert step_result.mtm_record.hedge_trade_cost == pytest.approx(0.05)
    assert step_result.mtm_record.hedge_turnover == pytest.approx(0.5)
    assert step_result.mtm_record.hedge_trade_count == 1
    assert step_result.mtm_record.hedge_pnl == pytest.approx(-0.05)


def test_mark_position_supports_custom_option_execution_model():
    class _FixedFillOptionExecutionModel:
        def execute(
            self,
            *,
            order,
        ) -> OptionExecutionResult:
            _ = order
            price_cost = 4.0 if order.trade_side > 0 else 0.0
            return OptionExecutionResult(
                fill_price=10.0,
                total_cost=price_cost,
                price_cost=price_cost,
                fee_cost=0.0,
            )

    setup = _make_setup(contracts=1, entry_price=5.0)
    cfg = _make_cfg()
    engine = _make_engine(
        rebalance_period=1,
        option_execution_model=_FixedFillOptionExecutionModel(),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            bid_price=6.0,
            ask_price=6.0,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    assert step_result.position is None
    assert len(step_result.trade_rows) == 1
    # Short entry at 5.0, mark PnL=-1.0 and buy-side exit cost=4.0 => net -5.0.
    assert step_result.trade_rows[0].pnl == pytest.approx(-5.0)
    assert step_result.mtm_record.delta_pnl == pytest.approx(-5.0)


def test_delta_hedge_policy_rejects_center_rebalance_for_ww_band():
    with pytest.raises(
        ValueError,
        match="WWDeltaBandModel requires rebalance_to='nearest_boundary'",
    ):
        DeltaHedgePolicy(
            enabled=True,
            trigger=HedgeTriggerPolicy(
                band_model=WWDeltaBandModel(),
            ),
            rebalance_to="center",
        )


def test_mark_position_ww_band_rebalances_to_nearest_boundary():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg(hedge_fee_bps=50.0)
    ww_band = WWDeltaBandModel(
        calibration_c=1.0,
        min_band_abs=0.0,
        max_band_abs=1.0,
    )
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=ww_band,
                rebalance_every_n_days=None,
            ),
            rebalance_to="nearest_boundary",
        ),
        hedge_market=_make_hedge_market(
            {"2020-01-02": 100.0},
            bid_prices={"2020-01-02": 100.0},
            ask_prices={"2020-01-02": 100.0},
        ),
        hedge_execution_model=FixedBpsHedgeExecutionModel(
            slip_ask=0.0,
            slip_bid=0.0,
            fee_bps=10.0,
        ),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=100.0,
            market_iv=0.2,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    fee_rate = 10.0 / 10_000.0
    expected_band = (fee_rate / (0.1 * 0.2 * 100.0 * 100.0)) ** (1.0 / 3.0)
    assert step_result.mtm_record.net_delta == pytest.approx(-expected_band)
    assert step_result.mtm_record.hedge_qty == pytest.approx(0.5 - expected_band)
    assert step_result.mtm_record.hedge_trade_count == 1


def test_mark_position_ww_band_skips_trade_inside_band():
    setup = _make_setup(contracts=1)
    cfg = _make_cfg(hedge_fee_bps=10.0)
    ww_band = WWDeltaBandModel(
        calibration_c=100.0,
        min_band_abs=0.0,
        max_band_abs=5.0,
    )
    engine = _make_engine(
        rebalance_period=10,
        delta_hedge_policy=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=ww_band,
                rebalance_every_n_days=None,
            ),
            rebalance_to="nearest_boundary",
        ),
        hedge_market=_make_hedge_market({"2020-01-02": 100.0}),
    )
    position, _ = engine.open_position(
        setup=setup,
        cfg=cfg,
        equity_running=10_000.0,
    )

    step_result = engine.mark_position(
        position=position,
        curr_date=pd.Timestamp("2020-01-02"),
        options=_make_options_row_for_date(
            trade_date="2020-01-02",
            spot_price=100.0,
            market_iv=0.2,
        ),
        cfg=cfg,
        equity_running=10_000.0,
    )

    fee_rate = 10.0 / 10_000.0
    expected_band = 100.0 * (fee_rate / (0.1 * 0.2 * 100.0 * 100.0)) ** (1.0 / 3.0)
    assert expected_band > 0.5
    assert step_result.mtm_record.hedge_qty == pytest.approx(0.0)
    assert step_result.mtm_record.net_delta == pytest.approx(-0.5)
    assert step_result.mtm_record.hedge_trade_count == 0
