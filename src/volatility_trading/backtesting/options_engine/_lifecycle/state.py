"""Lifecycle state contracts for options position execution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin import MarginAccount, MarginStatus

from ..types import EntryIntent


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
    last_spot: float
    last_iv: float
    last_delta: float
    last_gamma: float
    last_vega: float
    last_theta: float
    last_net_delta: float


@dataclass(frozen=True)
class EntryMarginSnapshot:
    """Entry-day margin/accounting snapshot used to build records/state."""

    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    initial_margin_requirement: float
    entry_delta_pnl: float
    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int


@dataclass(frozen=True)
class MarkValuationSnapshot:
    """One-date valuation snapshot before margin and exits are applied."""

    prev_mtm_before: float
    pnl_mtm: float
    delta: float
    gamma: float
    vega: float
    theta: float
    complete_leg_quotes: tuple[pd.Series, ...] | None
    has_missing_quote: bool
    spot: float
    iv: float
    hedge_pnl: float
    net_delta: float
    delta_pnl_market: float


@dataclass(frozen=True)
class MarkMarginSnapshot:
    """One-date margin/accounting state after evaluating the margin account."""

    initial_margin_requirement: float
    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int
    margin_status: MarginStatus | None


@dataclass(frozen=True, slots=True)
class MtmMarket:
    """Market marks captured in one lifecycle MTM record."""

    spot: float
    volatility: float


@dataclass(frozen=True, slots=True)
class MtmMargin:
    """Margin/accounting fields captured in one lifecycle MTM record."""

    per_contract: float | None
    initial_requirement: float
    maintenance_requirement: float
    excess: float
    deficit: float
    in_call: bool
    call_days: int
    forced_liquidation: bool
    contracts_liquidated: int
    financing_pnl: float


@dataclass(frozen=True, slots=True)
class MtmRecord:
    """One daily lifecycle mark-to-market ledger record."""

    date: pd.Timestamp
    market: MtmMarket
    delta_pnl: float
    delta: float
    net_delta: float
    gamma: float
    vega: float
    theta: float
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
            "delta": self.delta,
            "net_delta": self.net_delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "hedge_qty": self.hedge_qty,
            "hedge_price_prev": self.hedge_price_prev,
            "hedge_pnl": self.hedge_pnl,
            "open_contracts": self.open_contracts,
            "margin_per_contract": self.margin.per_contract,
            "initial_margin_requirement": self.margin.initial_requirement,
            "maintenance_margin_requirement": self.margin.maintenance_requirement,
            "margin_excess": self.margin.excess,
            "margin_deficit": self.margin.deficit,
            "in_margin_call": self.margin.in_call,
            "margin_call_days": self.margin.call_days,
            "forced_liquidation": self.margin.forced_liquidation,
            "contracts_liquidated": self.margin.contracts_liquidated,
            "financing_pnl": self.margin.financing_pnl,
        }
