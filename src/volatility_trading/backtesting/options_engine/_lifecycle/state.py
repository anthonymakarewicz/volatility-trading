"""Lifecycle state contracts for options position execution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount, MarginStatus
from volatility_trading.backtesting.types import BacktestConfig, MarginCore
from volatility_trading.options.types import Greeks, MarketState

from ..types import EntryIntent

# TODO: When adding delta hedging, keep two dataclasses for options and hedger for storign
# nb of contracts, delta for eahc, entry price ...
# Cretae a psoiton that would have a OptionsPositon and HedgerPositon that store all info into a
# centralzied dataclass


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
    """Mutable open-position state updated once per trading date.

    This object stores lifecycle state required to mark, hedge, finance, and
    close a position consistently across dates.
    """

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
    complete_leg_quotes: tuple[pd.Series, ...] | None
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
class MtmMargin:
    """Margin/accounting fields captured in one lifecycle MTM record."""

    per_contract: float | None
    initial_requirement: float
    core: MarginCore


@dataclass(frozen=True, slots=True)
class MtmRecord:
    """One daily lifecycle mark-to-market ledger record."""

    date: pd.Timestamp
    market: MarketState
    delta_pnl: float
    net_delta: float
    greeks: Greeks
    hedge_qty: float
    hedge_price_prev: float
    hedge_pnl: float
    open_contracts: int
    margin: MtmMargin

    def to_dict(self) -> dict[str, object]:
        """Flatten the MTM record into the canonical tabular row schema."""
        return {
            "date": self.date,
            "S": self.market.spot,
            "iv": self.market.volatility,
            "delta_pnl": self.delta_pnl,
            "delta": self.greeks.delta,
            "net_delta": self.net_delta,
            "gamma": self.greeks.gamma,
            "vega": self.greeks.vega,
            "theta": self.greeks.theta,
            "hedge_qty": self.hedge_qty,
            "hedge_price_prev": self.hedge_price_prev,
            "hedge_pnl": self.hedge_pnl,
            "open_contracts": self.open_contracts,
            "margin_per_contract": self.margin.per_contract,
            "initial_margin_requirement": self.margin.initial_requirement,
            "maintenance_margin_requirement": self.margin.core.maintenance_margin_requirement,
            "margin_excess": self.margin.core.margin_excess,
            "margin_deficit": self.margin.core.margin_deficit,
            "in_margin_call": self.margin.core.in_margin_call,
            "margin_call_days": self.margin.core.margin_call_days,
            "forced_liquidation": self.margin.core.forced_liquidation,
            "contracts_liquidated": self.margin.core.contracts_liquidated,
            "financing_pnl": self.margin.core.financing_pnl,
        }


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One realized trade ledger row emitted by lifecycle exits."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_dte: int | None
    expiry_date: pd.Timestamp | None
    contracts: int
    pnl: float
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_per_contract: float | None
    exit_type: str
    trade_legs: list[dict[str, object]]  # TODO: Weird this object type ?

    def to_dict(self) -> dict[str, object]:
        """Flatten trade record into the canonical trades table row."""
        return {
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_dte": self.entry_dte,
            "expiry_date": self.expiry_date,
            "contracts": self.contracts,
            "pnl": self.pnl,
            "risk_per_contract": self.risk_per_contract,
            "risk_worst_scenario": self.risk_worst_scenario,
            "margin_per_contract": self.margin_per_contract,
            "exit_type": self.exit_type,
            "trade_legs": self.trade_legs,
        }


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
