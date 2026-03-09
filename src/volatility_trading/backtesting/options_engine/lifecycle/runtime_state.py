"""Runtime lifecycle state contracts for options position execution.

These dataclasses model internal one-step snapshots used by open/mark/close
lifecycle helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from volatility_trading.backtesting.config import BacktestRunConfig
from volatility_trading.backtesting.margin import MarginAccount, MarginStatus
from volatility_trading.backtesting.margin_types import MarginCore
from volatility_trading.options.types import Greeks, MarketState

from ..contracts.market import QuoteSnapshot


@dataclass(frozen=True)
class EntryMarginSnapshot:
    """Entry-day margin/accounting snapshot used to build records/state."""

    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    initial_margin_requirement: float
    option_trade_cost: float
    entry_delta_pnl: float
    margin: MarginCore


@dataclass(frozen=True, slots=True)
class HedgeTelemetry:
    """Per-step hedge telemetry used for diagnostics and aggregation."""

    carry_pnl: float = 0.0
    trade_cost: float = 0.0
    turnover: float = 0.0
    trade_count: int = 0


@dataclass(frozen=True, slots=True)
class HedgeValuation:
    """Per-step hedge valuation and telemetry outputs."""

    price_prev: float
    pnl: float
    telemetry: HedgeTelemetry = field(default_factory=HedgeTelemetry)


@dataclass(frozen=True)
class MarkValuationSnapshot:
    """One-date valuation snapshot before margin and exits are applied."""

    prev_mtm_before: float
    pnl_mtm: float
    greeks: Greeks
    complete_leg_quotes: tuple[QuoteSnapshot, ...] | None
    has_missing_quote: bool
    market: MarketState
    hedge: HedgeValuation
    net_delta: float
    option_market_pnl: float
    delta_pnl_market: float


@dataclass(frozen=True)
class HedgeStepSnapshot:
    """One-date hedge step result applied on top of option valuation."""

    hedge: HedgeValuation
    net_delta: float
    target_net_delta: float
    trade_qty: float


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
    cfg: BacktestRunConfig
    equity_running: float
