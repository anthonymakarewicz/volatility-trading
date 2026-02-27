from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import pandas as pd

# --- Shared aliases -------------------------------------------------
DataMapping: TypeAlias = Mapping[str, Any]

# --- Core dataclasses -----------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    leverage: float = 1.0

    # Execution / market microstructure
    lot_size: int = 100
    hedge_size: int = 50
    slip_ask: float = 0.01
    slip_bid: float = 0.01
    commission_per_leg: float = 1.0

    # Risk “floors” that are environment-like
    risk_pc_floor: float = 750.0


class LifecycleStepLike(Protocol):
    """Runtime shape required by the backtest kernel mark step."""

    position: Any | None
    mtm_record: Any
    trade_rows: list[Any]


MarkOpenPositionFn: TypeAlias = Callable[[Any, pd.Timestamp, float], LifecycleStepLike]
PrepareEntryFn: TypeAlias = Callable[[pd.Timestamp, float], Any | None]
OpenPositionFn: TypeAlias = Callable[[Any, float], tuple[Any, Any]]
CanReenterFn: TypeAlias = Callable[[list[Any]], bool]
BuildOutputsFn: TypeAlias = Callable[[list[Any], list[Any], float], tuple[Any, Any]]


@dataclass(frozen=True)
class BacktestKernelHooks:
    """Callback bundle consumed by the engine-owned single-position loop."""

    mark_open_position: MarkOpenPositionFn
    prepare_entry: PrepareEntryFn
    open_position: OpenPositionFn
    can_reenter_same_day: CanReenterFn


@dataclass(frozen=True)
class BacktestExecutionPlan:
    """Engine-executable simulation plan compiled by one strategy runner."""

    trading_dates: list[pd.Timestamp]
    active_signal_dates: set[pd.Timestamp]
    initial_equity: float
    hooks: BacktestKernelHooks
    build_outputs: BuildOutputsFn


@dataclass(frozen=True, slots=True)
class MarginCore:
    """Shared margin/accounting state used across backtesting lifecycle snapshots."""

    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int

    @classmethod
    def empty(cls) -> MarginCore:
        """Return default non-margin-call state used before account evaluation."""
        return cls(
            financing_pnl=0.0,
            maintenance_margin_requirement=0.0,
            margin_excess=float("nan"),
            margin_deficit=float("nan"),
            in_margin_call=False,
            margin_call_days=0,
            forced_liquidation=False,
            contracts_liquidated=0,
        )
