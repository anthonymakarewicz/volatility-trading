"""Options-specific execution-kernel contracts consumed by ``Backtester``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ._lifecycle.ledger import MtmRecord, TradeRecord
from ._lifecycle.runtime_state import (
    LifecycleStepResult,
    OpenPosition,
    PositionEntrySetup,
)

MarkOpenPositionFn = Callable[[OpenPosition, pd.Timestamp, float], LifecycleStepResult]
PrepareEntryFn = Callable[[pd.Timestamp, float], PositionEntrySetup | None]
OpenPositionFn = Callable[[PositionEntrySetup, float], tuple[OpenPosition, MtmRecord]]
CanReenterFn = Callable[[list[TradeRecord]], bool]
BuildOutputsFn = Callable[
    [list[TradeRecord], list[MtmRecord], float],
    tuple[pd.DataFrame, pd.DataFrame],
]


@dataclass(frozen=True)
class BacktestKernelHooks:
    """Callbacks consumed by the engine-owned single-position loop."""

    mark_open_position: MarkOpenPositionFn
    prepare_entry: PrepareEntryFn
    open_position: OpenPositionFn
    can_reenter_same_day: CanReenterFn


@dataclass(frozen=True)
class BacktestExecutionPlan:
    """Executable options simulation plan compiled from one strategy spec."""

    trading_dates: list[pd.Timestamp]
    active_signal_dates: set[pd.Timestamp]
    initial_equity: float
    hooks: BacktestKernelHooks
    build_outputs: BuildOutputsFn
