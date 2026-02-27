from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting.engine import run_backtest_execution_plan
from volatility_trading.backtesting.options_engine.contracts import (
    SinglePositionExecutionPlan,
    SinglePositionHooks,
)
from volatility_trading.backtesting.options_engine.records import (
    MtmMargin,
    MtmRecord,
    TradeRecord,
)
from volatility_trading.backtesting.options_engine.state import (
    LifecycleStepResult,
    OpenPosition,
    PositionEntrySetup,
)
from volatility_trading.backtesting.options_engine.types import (
    EntryIntent,
    LegSelection,
    LegSpec,
    QuoteSnapshot,
)
from volatility_trading.backtesting.types import (
    MarginCore,
)
from volatility_trading.options import Greeks, MarketState, OptionType, PositionSide


def _make_entry_setup(entry_date: pd.Timestamp) -> PositionEntrySetup:
    leg_spec = LegSpec(option_type=OptionType.CALL, delta_target=0.5)
    leg_quote = QuoteSnapshot(
        option_type_label="C",
        strike=100.0,
        bid_price=1.0,
        ask_price=1.1,
        delta=0.5,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        expiry_date=pd.Timestamp("2020-01-31"),
        dte=30,
        spot_price=100.0,
        smoothed_iv=0.2,
    )
    leg = LegSelection(
        spec=leg_spec,
        quote=leg_quote,
        side=PositionSide.LONG,
        entry_price=1.1,
    )
    intent = EntryIntent(
        entry_date=entry_date,
        expiry_date=pd.Timestamp("2020-01-31"),
        chosen_dte=30,
        legs=(leg,),
        entry_state=MarketState(spot=100.0, volatility=0.2),
    )
    return PositionEntrySetup(
        intent=intent,
        contracts=1,
        risk_per_contract=None,
        risk_worst_scenario=None,
        margin_per_contract=None,
    )


def _make_open_position(setup: PositionEntrySetup) -> OpenPosition:
    return OpenPosition(
        entry_date=setup.intent.entry_date,
        expiry_date=setup.intent.expiry_date,
        chosen_dte=setup.intent.chosen_dte,
        rebalance_date=None,
        max_hold_date=None,
        intent=setup.intent,
        contracts_open=setup.contracts,
        risk_per_contract=setup.risk_per_contract,
        risk_worst_scenario=setup.risk_worst_scenario,
        margin_account=None,
        latest_margin_per_contract=setup.margin_per_contract,
        net_entry=0.0,
        prev_mtm=0.0,
        hedge_qty=0.0,
        hedge_price_entry=float("nan"),
        last_market=setup.intent.entry_state or MarketState(spot=100.0, volatility=0.2),
        last_greeks=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
        last_net_delta=0.0,
    )


def _make_mtm_record(*, date: pd.Timestamp, delta_pnl: float) -> MtmRecord:
    return MtmRecord(
        date=date,
        market=MarketState(spot=100.0, volatility=0.2),
        delta_pnl=float(delta_pnl),
        net_delta=0.0,
        greeks=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
        hedge_qty=0.0,
        hedge_price_prev=float("nan"),
        hedge_pnl=0.0,
        open_contracts=1,
        margin=MtmMargin(
            per_contract=None,
            initial_requirement=0.0,
            core=MarginCore.empty(),
        ),
    )


def _make_trade_record(*, exit_type: str) -> TradeRecord:
    return TradeRecord(
        entry_date=pd.Timestamp("2020-01-01"),
        exit_date=pd.Timestamp("2020-01-02"),
        entry_dte=30,
        expiry_date=pd.Timestamp("2020-01-31"),
        contracts=1,
        pnl=0.0,
        risk_per_contract=None,
        risk_worst_scenario=None,
        margin_per_contract=None,
        exit_type=exit_type,
        trade_legs=(),
    )


def _build_test_outputs(
    trades: list[TradeRecord],
    mtm: list[MtmRecord],
    _initial_equity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame({"exit_type": [trade.exit_type for trade in trades]}),
        pd.DataFrame({"delta_pnl": [record.delta_pnl for record in mtm]}),
    )


def _plan(
    *,
    trading_dates: list[pd.Timestamp],
    active_signal_dates: set[pd.Timestamp],
    hooks: SinglePositionHooks,
    initial_equity: float = 100.0,
) -> SinglePositionExecutionPlan:
    return SinglePositionExecutionPlan(
        trading_dates=trading_dates,
        active_signal_dates=active_signal_dates,
        initial_equity=float(initial_equity),
        hooks=hooks,
        build_outputs=_build_test_outputs,
    )


def test_kernel_blocks_same_day_reentry_when_policy_disallows():
    trading_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    active_signal_dates = set(trading_dates)
    entry_count = 0

    def prepare_entry(curr_date: pd.Timestamp, equity: float):
        _ = (curr_date, equity)
        return _make_entry_setup(curr_date)

    def open_position(setup: PositionEntrySetup, equity: float):
        nonlocal entry_count
        _ = equity
        entry_count += 1
        return _make_open_position(setup), _make_mtm_record(
            date=setup.intent.entry_date,
            delta_pnl=0.0,
        )

    def mark_open_position(
        position: OpenPosition,
        curr_date: pd.Timestamp,
        equity: float,
    ) -> LifecycleStepResult:
        _ = (position, equity)
        return LifecycleStepResult(
            position=None,
            mtm_record=_make_mtm_record(date=curr_date, delta_pnl=1.0),
            trade_rows=[_make_trade_record(exit_type="Rebalance Period")],
        )

    hooks = SinglePositionHooks(
        mark_open_position=mark_open_position,
        prepare_entry=prepare_entry,
        open_position=open_position,
        can_reenter_same_day=lambda rows: False,
    )
    trades, mtm = run_backtest_execution_plan(
        _plan(
            trading_dates=trading_dates,
            active_signal_dates=active_signal_dates,
            hooks=hooks,
        )
    )

    assert len(trades) == 1
    assert len(mtm) == 2
    assert entry_count == 1


def test_kernel_can_reenter_same_day_and_uses_updated_equity():
    trading_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    active_signal_dates = set(trading_dates)
    prepare_equities: list[float] = []
    entry_count = 0

    def prepare_entry(curr_date: pd.Timestamp, equity: float):
        prepare_equities.append(float(equity))
        return _make_entry_setup(curr_date)

    def open_position(setup: PositionEntrySetup, equity: float):
        nonlocal entry_count
        _ = equity
        entry_count += 1
        return _make_open_position(setup), _make_mtm_record(
            date=setup.intent.entry_date,
            delta_pnl=-2.0,
        )

    def mark_open_position(
        position: OpenPosition,
        curr_date: pd.Timestamp,
        equity: float,
    ) -> LifecycleStepResult:
        _ = (position, equity)
        return LifecycleStepResult(
            position=None,
            mtm_record=_make_mtm_record(date=curr_date, delta_pnl=5.0),
            trade_rows=[_make_trade_record(exit_type="Rebalance Period")],
        )

    hooks = SinglePositionHooks(
        mark_open_position=mark_open_position,
        prepare_entry=prepare_entry,
        open_position=open_position,
        can_reenter_same_day=lambda rows: True,
    )
    trades, mtm = run_backtest_execution_plan(
        _plan(
            trading_dates=trading_dates,
            active_signal_dates=active_signal_dates,
            hooks=hooks,
        )
    )

    assert len(trades) == 1
    assert len(mtm) == 3
    assert entry_count == 2
    assert prepare_equities == [100.0, 103.0]
