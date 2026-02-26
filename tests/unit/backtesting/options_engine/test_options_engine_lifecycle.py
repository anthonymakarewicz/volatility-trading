import pandas as pd
import pytest

from volatility_trading.backtesting import BacktestConfig, MarginPolicy
from volatility_trading.backtesting.options_engine import (
    EntryIntent,
    ExitRuleSet,
    LegSelection,
    LegSpec,
    PositionEntrySetup,
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


def _make_cfg(*, commission_per_leg: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=10_000.0,
        lot_size=1,
        slip_ask=0.0,
        slip_bid=0.0,
        commission_per_leg=commission_per_leg,
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
    smoothed_iv: float = 0.2,
) -> pd.Series:
    return pd.Series(
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
            "smoothed_iv": smoothed_iv,
        }
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


def _make_options_row_for_date(
    *,
    trade_date: str = "2020-01-02",
    strike: float = 100.0,
    option_type: str = "C",
    bid_price: float = 5.0,
    ask_price: float = 5.0,
    spot_price: float = 101.0,
    smoothed_iv: float = 0.21,
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
        "smoothed_iv": smoothed_iv,
    }
    options = pd.DataFrame([row])
    options["trade_date"] = pd.to_datetime(options["trade_date"])
    options["expiry_date"] = pd.to_datetime(options["expiry_date"])
    return options.set_index("trade_date").sort_index()


def _make_engine(
    *,
    rebalance_period: int | None = 5,
    margin_policy: MarginPolicy | None = None,
    margin_model=None,
) -> PositionLifecycleEngine:
    return PositionLifecycleEngine(
        rebalance_period=rebalance_period,
        max_holding_period=None,
        exit_rule_set=ExitRuleSet.period_rules(),
        margin_policy=margin_policy,
        margin_model=margin_model,
        pricer=_NullPricer(),
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
    assert entry_record.delta_pnl == pytest.approx(-6.0)
    assert entry_record.greeks.delta == pytest.approx(-1.5)
    assert entry_record.net_delta == pytest.approx(-1.5)
    assert entry_record.greeks.gamma == pytest.approx(-0.3)
    assert entry_record.greeks.vega == pytest.approx(-0.6)
    assert entry_record.greeks.theta == pytest.approx(0.9)


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
    assert trade_rows[0]["exit_type"] == "Rebalance Period"
    assert trade_rows[0]["contracts"] == 2
    assert trade_rows[0]["pnl"] == pytest.approx(-2.0)
    assert mtm_record.delta_pnl == pytest.approx(-2.0)
    assert mtm_record.open_contracts == 0
    assert mtm_record.greeks.delta == pytest.approx(0.0)
    assert mtm_record.net_delta == pytest.approx(0.0)


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
    assert trade_rows[0]["exit_type"] == "Margin Call Liquidation"
    assert trade_rows[0]["contracts"] == 3
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
    assert trade_rows[0]["exit_type"] == "Margin Call Partial Liquidation"
    assert trade_rows[0]["contracts"] == 2
    assert updated_position.contracts_open == 2
    assert mtm_record.open_contracts == 2
    assert mtm_record.margin.core.contracts_liquidated == 2
    assert mtm_record.margin.core.forced_liquidation is True
