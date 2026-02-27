from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

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
