"""Public runtime state dataclasses for options execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount
from volatility_trading.options.types import Greeks, MarketState

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
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    net_entry: float
    prev_mtm: float
    hedge_qty: float
    hedge_price_entry: float
    last_market: MarketState
    last_greeks: Greeks
    last_net_delta: float


@dataclass(frozen=True, slots=True)
class LifecycleStepResult:
    """Normalized one-step lifecycle outcome returned by mark handlers."""

    position: OpenPosition | None
    mtm_record: MtmRecord
    trade_rows: list[TradeRecord]
