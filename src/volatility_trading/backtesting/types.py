from dataclasses import dataclass
from typing import Any, Mapping

# --- Shared aliases -------------------------------------------------
DataMapping = Mapping[str, Any]
ParamGrid   = dict[str, Any]

# --- Core dataclasses -----------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    leverage: float        = 1.0

    # Execution / market microstructure
    lot_size: int          = 100
    hedge_size: int        = 50
    slip_ask: float        = 0.01
    slip_bid: float        = 0.01
    commission_per_leg: float = 1.0

    # Risk “floors” that are environment-like
    risk_pc_floor: float   = 750.0


@dataclass
class SliceContext:
    data: Mapping[str, Any]   # {"options": df, "features": df, "hedge": series, ...}
    params: dict[str, Any]
    config: BacktestConfig
    capital: float            # current capital for this run / window