import pandas as pd

from volatility_trading.strategies.options_core import (
    SinglePositionRunnerHooks,
    run_single_position_date_loop,
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
        return {"position_id": entry_count}, {
            "date": trading_dates[0],
            "delta_pnl": 0.0,
        }

    def mark_open_position(position: dict, curr_date: pd.Timestamp, equity: float):
        _ = (position, curr_date, equity)
        return (
            None,
            {"date": curr_date, "delta_pnl": 1.0},
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
        return {"position_id": entry_count}, {
            "date": trading_dates[0],
            "delta_pnl": -2.0,
        }

    def mark_open_position(position: dict, curr_date: pd.Timestamp, equity: float):
        _ = (position, curr_date, equity)
        return (
            None,
            {"date": curr_date, "delta_pnl": 5.0},
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
