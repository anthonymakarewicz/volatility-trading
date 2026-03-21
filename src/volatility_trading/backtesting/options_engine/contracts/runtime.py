"""Public runtime state dataclasses for options execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount
from volatility_trading.options.risk.types import StressPoint
from volatility_trading.options.types import Greeks, MarketState

from ..factor_models import FactorSnapshot
from .records import MtmRecord, TradeRecord
from .structures import EntryIntent


@dataclass(frozen=True)
class PositionEntrySetup:
    """Entry payload consumed by the lifecycle engine."""

    intent: EntryIntent
    contracts: int
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_per_contract: float | None
    entry_stress_points: tuple[StressPoint, ...] = ()
    risk_budget_contracts: int | None = None
    margin_budget_contracts: int | None = None
    sizing_binding_constraint: str | None = None
    min_contracts_override_applied: bool = False


@dataclass
class HedgeState:
    """Mutable hedge state carried by one open option position."""

    qty: float = 0.0
    last_price: float = float("nan")
    last_rebalance_date: pd.Timestamp | None = None


@dataclass
class OpenPosition:
    """Mutable open-position state updated once per trading date."""

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp | None
    chosen_dte: int | None
    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None
    intent: EntryIntent
    contracts_open: int
    option_contract_multiplier: float
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    net_entry: float
    entry_option_trade_cost: float
    prev_mtm: float
    hedge: HedgeState
    last_market: MarketState
    last_greeks: Greeks
    last_net_delta: float
    last_factor_snapshot: FactorSnapshot = field(default_factory=FactorSnapshot)
    entry_stress_points: tuple[StressPoint, ...] = field(default_factory=tuple)
    risk_budget_contracts: int | None = None
    margin_budget_contracts: int | None = None
    sizing_binding_constraint: str | None = None
    min_contracts_override_applied: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleStepResult:
    """Normalized one-step lifecycle outcome returned by mark handlers."""

    position: OpenPosition | None
    mtm_record: MtmRecord
    trade_rows: list[TradeRecord]
