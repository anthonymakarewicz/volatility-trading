"""Runtime lifecycle state contracts for options position execution.

These dataclasses model in-flight simulation state and per-step snapshots used
by open/mark/close lifecycle handlers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount, MarginStatus
from volatility_trading.backtesting.types import BacktestConfig, MarginCore
from volatility_trading.options.types import Greeks, MarketState

from ..types import EntryIntent, QuoteSnapshot
from .ledger import MtmRecord, TradeRecord


@dataclass(frozen=True)
class PositionEntrySetup:
    """Entry payload consumed by the generic lifecycle engine.

    Attributes:
        intent: Resolved structure entry intent.
        contracts: Number of structure contracts to open.
        risk_per_contract: Optional worst-case risk estimate used for sizing.
        risk_worst_scenario: Optional label for the worst risk scenario.
        margin_per_contract: Optional initial margin estimate per contract.
    """

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


@dataclass(frozen=True)
class EntryMarginSnapshot:
    """Entry-day margin/accounting snapshot used to build records/state."""

    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    initial_margin_requirement: float
    entry_delta_pnl: float
    margin: MarginCore


@dataclass(frozen=True)
class MarkValuationSnapshot:
    """One-date valuation snapshot before margin and exits are applied."""

    prev_mtm_before: float
    pnl_mtm: float
    greeks: Greeks
    complete_leg_quotes: tuple[QuoteSnapshot, ...] | None
    has_missing_quote: bool
    market: MarketState
    hedge_pnl: float
    net_delta: float
    delta_pnl_market: float


@dataclass(frozen=True)
class MarkMarginSnapshot:
    """One-date margin/accounting state after evaluating the margin account."""

    initial_margin_requirement: float
    margin: MarginCore
    margin_status: MarginStatus | None


@dataclass(frozen=True, slots=True)
class LifecycleStepContext:
    """Shared one-date context passed across lifecycle mark/exit handlers."""

    curr_date: pd.Timestamp
    cfg: BacktestConfig
    equity_running: float
    lot_size: int
    roundtrip_commission_per_contract: float


@dataclass(frozen=True, slots=True)
class LifecycleStepResult:
    """Normalized one-step lifecycle outcome returned by mark handlers."""

    position: OpenPosition | None
    mtm_record: MtmRecord
    trade_rows: list[TradeRecord]
