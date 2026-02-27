"""Execution-plan contracts consumed by the engine-owned runtime loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .records import MtmRecord, TradeRecord
from .runtime import LifecycleStepResult, OpenPosition, PositionEntrySetup

SinglePositionMarkFn = Callable[
    [OpenPosition, pd.Timestamp, float], LifecycleStepResult
]
SinglePositionPrepareEntryFn = Callable[
    [pd.Timestamp, float], PositionEntrySetup | None
]
SinglePositionOpenFn = Callable[
    [PositionEntrySetup, float], tuple[OpenPosition, MtmRecord]
]
SinglePositionCanReenterFn = Callable[[list[TradeRecord]], bool]
SinglePositionBuildOutputsFn = Callable[
    [list[TradeRecord], list[MtmRecord], float],
    tuple[pd.DataFrame, pd.DataFrame],
]


@dataclass(frozen=True)
class SinglePositionHooks:
    """Callbacks consumed by the engine-owned single-position loop."""

    mark_open_position: SinglePositionMarkFn
    prepare_entry: SinglePositionPrepareEntryFn
    open_position: SinglePositionOpenFn
    can_reenter_same_day: SinglePositionCanReenterFn


@dataclass(frozen=True)
class SinglePositionExecutionPlan:
    """Executable options simulation plan compiled from one strategy spec."""

    trading_dates: list[pd.Timestamp]
    active_signal_dates: set[pd.Timestamp]
    initial_equity: float
    hooks: SinglePositionHooks
    build_outputs: SinglePositionBuildOutputsFn
