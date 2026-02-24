from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

# --- Shared aliases -------------------------------------------------
DataMapping: TypeAlias = Mapping[str, Any]
ParamGrid: TypeAlias = dict[str, Any]

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


@dataclass
class SliceContext:
    data: Mapping[str, Any]  # {"options": df, "features": df, "hedge": series, ...}
    params: dict[str, Any]
    config: BacktestConfig
    capital: float  # current capital for this run / window


class StrategyRunner(Protocol):
    """Minimal runtime contract for backtestable strategies."""

    def run(self, ctx: SliceContext) -> tuple[Any, Any]:
        """Execute strategy and return `(trades, mtm)` tabular outputs."""
