import pandas as pd

from volatility_trading.backtesting.options_engine import (
    SinglePositionRunnerHooks,
    run_single_position_date_loop,
)
from volatility_trading.backtesting.options_engine._lifecycle.state import (
    MtmMargin,
    MtmRecord,
)
from volatility_trading.backtesting.types import MarginCore
from volatility_trading.options.types import Greeks, MarketState


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


def test_runner_blocks_same_day_reentry_when_policy_disallows():
    trading_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    active_signal_dates = set(trading_dates)
    entry_count = 0

    def prepare_entry(curr_date: pd.Timestamp, equity: float):
        _ = (curr_date, equity)
        return {"ok": True}

    def open_position(setup: dict, equity: float):
        nonlocal entry_count
        _ = (setup, equity)
        entry_count += 1
        return {"position_id": entry_count}, _make_mtm_record(
            date=trading_dates[0],
            delta_pnl=0.0,
        )

    def mark_open_position(position: dict, curr_date: pd.Timestamp, equity: float):
        _ = (position, curr_date, equity)
        return (
            None,
            _make_mtm_record(date=curr_date, delta_pnl=1.0),
            [{"exit_type": "Rebalance Period"}],
        )

    hooks = SinglePositionRunnerHooks[dict, dict](
        mark_open_position=mark_open_position,
        prepare_entry=prepare_entry,
        open_position=open_position,
        can_reenter_same_day=lambda rows: False,
    )
    trades, mtm_records = run_single_position_date_loop(
        trading_dates=trading_dates,
        active_signal_dates=active_signal_dates,
        initial_equity=100.0,
        hooks=hooks,
    )

    assert len(trades) == 1
    assert len(mtm_records) == 2
    assert entry_count == 1


def test_runner_can_reenter_same_day_and_uses_updated_equity():
    trading_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    active_signal_dates = set(trading_dates)
    prepare_equities: list[float] = []
    entry_count = 0

    def prepare_entry(curr_date: pd.Timestamp, equity: float):
        _ = curr_date
        prepare_equities.append(float(equity))
        return {"ok": True}

    def open_position(setup: dict, equity: float):
        nonlocal entry_count
        _ = (setup, equity)
        entry_count += 1
        return {"position_id": entry_count}, _make_mtm_record(
            date=trading_dates[0],
            delta_pnl=-2.0,
        )

    def mark_open_position(position: dict, curr_date: pd.Timestamp, equity: float):
        _ = (position, curr_date, equity)
        return (
            None,
            _make_mtm_record(date=curr_date, delta_pnl=5.0),
            [{"exit_type": "Rebalance Period"}],
        )

    hooks = SinglePositionRunnerHooks[dict, dict](
        mark_open_position=mark_open_position,
        prepare_entry=prepare_entry,
        open_position=open_position,
        can_reenter_same_day=lambda rows: True,
    )
    trades, mtm_records = run_single_position_date_loop(
        trading_dates=trading_dates,
        active_signal_dates=active_signal_dates,
        initial_equity=100.0,
        hooks=hooks,
    )

    assert len(trades) == 1
    assert len(mtm_records) == 3
    assert entry_count == 2
    assert prepare_equities == [100.0, 103.0]
