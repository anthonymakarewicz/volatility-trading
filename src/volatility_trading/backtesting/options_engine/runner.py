"""Shared event loop for single-position, date-driven options strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import pandas as pd

PositionT = TypeVar("PositionT")
SetupT = TypeVar("SetupT")

# TODO: Extend to multiple positions


@dataclass(frozen=True)
class SinglePositionRunnerHooks(Generic[PositionT, SetupT]):
    """Callback bundle required by the single-position event loop.

    Attributes:
        mark_open_position: Revalue open position and optionally close it.
        prepare_entry: Build an entry setup for a trade date and current equity.
        open_position: Open a new position and emit entry-day MTM record.
        can_reenter_same_day: Decide if immediate reentry is allowed after exits.
    """

    mark_open_position: Callable[
        [PositionT, pd.Timestamp, float], tuple[PositionT | None, dict, list[dict]]
    ]
    prepare_entry: Callable[[pd.Timestamp, float], SetupT | None]
    open_position: Callable[[SetupT, float], tuple[PositionT, dict]]
    can_reenter_same_day: Callable[[list[dict]], bool]


def run_single_position_date_loop(
    *,
    trading_dates: list[pd.Timestamp],
    active_signal_dates: set[pd.Timestamp],
    initial_equity: float,
    hooks: SinglePositionRunnerHooks[PositionT, SetupT],
) -> tuple[list[dict], list[dict]]:
    """Run the shared single-position event loop.

    Returns:
        A tuple ``(trades, mtm_records)`` built from daily lifecycle callbacks.
    """
    trades: list[dict] = []
    mtm_records: list[dict] = []
    equity_running = float(initial_equity)
    open_position: PositionT | None = None

    for curr_date in trading_dates:
        if open_position is not None:
            open_position, mtm_record, trade_rows = hooks.mark_open_position(
                open_position,
                curr_date,
                equity_running,
            )
            mtm_records.append(mtm_record)
            trades.extend(trade_rows)
            equity_running += float(mtm_record["delta_pnl"])

            if open_position is not None:
                continue
            if not hooks.can_reenter_same_day(trade_rows):
                continue

        if curr_date not in active_signal_dates:
            continue

        setup = hooks.prepare_entry(curr_date, equity_running)
        if setup is None:
            continue

        open_position, entry_record = hooks.open_position(setup, equity_running)
        mtm_records.append(entry_record)
        equity_running += float(entry_record["delta_pnl"])

    return trades, mtm_records
