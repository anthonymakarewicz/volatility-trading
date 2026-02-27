"""Runtime lifecycle state contracts for options position execution.

These dataclasses model internal one-step snapshots used by open/mark/close
lifecycle helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount, MarginStatus
from volatility_trading.backtesting.types import BacktestConfig, MarginCore
from volatility_trading.options.types import Greeks, MarketState

from ..types import QuoteSnapshot


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
