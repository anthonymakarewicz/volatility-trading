"""Shared event loop for single-position, date-driven options strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import pandas as pd

from ._lifecycle.state import LifecycleStepResult, MtmRecord, TradeRecord

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
        [PositionT, pd.Timestamp, float],
        LifecycleStepResult,
    ]
    prepare_entry: Callable[[pd.Timestamp, float], SetupT | None]
    open_position: Callable[[SetupT, float], tuple[PositionT, MtmRecord]]
    can_reenter_same_day: Callable[[list[TradeRecord]], bool]


def _record_delta_pnl(record: MtmRecord) -> float:
    """Return ``delta_pnl`` from typed MTM records."""
    return float(record.delta_pnl)


def run_single_position_date_loop(
    *,
    trading_dates: list[pd.Timestamp],
    active_signal_dates: set[pd.Timestamp],
    initial_equity: float,
    hooks: SinglePositionRunnerHooks[PositionT, SetupT],
) -> tuple[list[TradeRecord], list[MtmRecord]]:
    """Run the shared single-position event loop.

    Returns:
        A tuple ``(trades, mtm_records)`` built from daily lifecycle callbacks.
    """
    trades: list[TradeRecord] = []
    mtm_records: list[MtmRecord] = []
    equity_running = float(initial_equity)
    open_position: PositionT | None = None

    for curr_date in trading_dates:
        if open_position is not None:
            step_result = hooks.mark_open_position(
                open_position,
                curr_date,
                equity_running,
            )
            open_position = step_result.position
            mtm_record = step_result.mtm_record
            trade_rows = step_result.trade_rows
            mtm_records.append(mtm_record)
            trades.extend(trade_rows)
            equity_running += _record_delta_pnl(mtm_record)

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
        equity_running += _record_delta_pnl(entry_record)

    return trades, mtm_records
