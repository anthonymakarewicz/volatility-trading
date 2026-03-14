"""Exit-rule framework and same-day reentry policy for options strategies.

This module isolates lifecycle close triggers from execution mechanics so
strategies can compose exit behavior without rewriting lifecycle code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from .contracts.records import TradeRecord


@runtime_checkable
class HasPeriodicExitDates(Protocol):
    """Minimal position contract required by periodic exit rules."""

    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None


@runtime_checkable
class ExitRule(Protocol):
    """Contract for one exit rule evaluated against an open position."""

    @property
    def exit_type(self) -> str:
        """Human-readable label emitted on trade close rows."""
        ...

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        """Return True when the rule requires position close on `curr_date`."""
        ...


@dataclass(frozen=True)
class ExitRuleState:
    """Minimal evaluation snapshot shared across exit rules.

    The richer-exit v1 surface exposes only structure-level
    ``pnl_per_contract`` so stop-loss / take-profit rules stay independent of
    account equity, total contracts, and optional risk-sizer plumbing.
    """

    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None
    pnl_per_contract: float | None = None

    @classmethod
    def from_position(
        cls,
        *,
        position: HasPeriodicExitDates,
        pnl_per_contract: float | None = None,
    ) -> ExitRuleState:
        """Clone the minimal rule inputs from one open-position object."""
        return cls(
            rebalance_date=position.rebalance_date,
            max_hold_date=position.max_hold_date,
            pnl_per_contract=(
                pnl_per_contract
                if pnl_per_contract is not None
                else getattr(position, "pnl_per_contract", None)
            ),
        )


@dataclass(frozen=True)
class RebalanceExitRule:
    """Exit rule based on rebalance date reached."""

    exit_type: str = "Rebalance Period"

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        return (
            position.rebalance_date is not None and curr_date >= position.rebalance_date
        )


@dataclass(frozen=True)
class MaxHoldingExitRule:
    """Exit rule based on max holding date reached."""

    exit_type: str = "Max Holding Period"

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        return (
            position.max_hold_date is not None and curr_date >= position.max_hold_date
        )


@dataclass(frozen=True)
class StopLossExitRule:
    """Exit when unrealized net P&L per contract breaches a loss threshold.

    The trigger uses lifecycle mark-step ``pnl_per_contract`` semantics:

    ``(valuation.pnl_mtm - entry_option_trade_cost + hedge.pnl) / contracts_open``

    This excludes hypothetical exit transaction costs because the rule is
    evaluated before an exit is executed.
    """

    threshold_per_contract: float
    exit_type: str = "Stop Loss"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_per_contract)
            or self.threshold_per_contract <= 0
        ):
            raise ValueError("threshold_per_contract must be > 0")

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        _ = curr_date
        pnl_per_contract = getattr(position, "pnl_per_contract", None)
        return pnl_per_contract is not None and pnl_per_contract <= -float(
            self.threshold_per_contract
        )


@dataclass(frozen=True)
class TakeProfitExitRule:
    """Exit when unrealized net P&L per contract reaches a profit target.

    The trigger uses lifecycle mark-step ``pnl_per_contract`` semantics:

    ``(valuation.pnl_mtm - entry_option_trade_cost + hedge.pnl) / contracts_open``

    This excludes hypothetical exit transaction costs because the rule is
    evaluated before an exit is executed.
    """

    threshold_per_contract: float
    exit_type: str = "Take Profit"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_per_contract)
            or self.threshold_per_contract <= 0
        ):
            raise ValueError("threshold_per_contract must be > 0")

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        _ = curr_date
        pnl_per_contract = getattr(position, "pnl_per_contract", None)
        return pnl_per_contract is not None and pnl_per_contract >= float(
            self.threshold_per_contract
        )


@dataclass(frozen=True)
class ExitRuleSet:
    """Set of exit rules evaluated together for one position/date."""

    rules: tuple[ExitRule, ...]
    combined_rebalance_max_hold_type: str = "Rebalance/Max Holding Period"

    @classmethod
    def no_rules(cls) -> ExitRuleSet:
        """Return an empty rule set for signal-driven lifecycle only."""
        return cls(rules=())

    @classmethod
    def period_rules(cls) -> ExitRuleSet:
        """Return default periodic exit rules (rebalance + max-holding)."""
        return cls(rules=(RebalanceExitRule(), MaxHoldingExitRule()))

    def evaluate(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
        pnl_per_contract: float | None = None,
    ) -> str | None:
        """Return exit-type label when one or more rules are triggered."""
        state = ExitRuleState.from_position(
            position=position,
            pnl_per_contract=pnl_per_contract,
        )
        triggered = tuple(
            rule.exit_type
            for rule in self.rules
            if rule.is_triggered(curr_date=curr_date, position=state)
        )
        if not triggered:
            return None
        if len(triggered) == 1:
            return triggered[0]
        if {"Rebalance Period", "Max Holding Period"}.issubset(set(triggered)):
            return self.combined_rebalance_max_hold_type
        return "/".join(triggered)


# TODO: Consider removing it adds nosie and will almsot never be tuned, mayeb add a reentry_after_n_days
# Maybe add a cooldown perido for reentries
@dataclass(frozen=True)
class SameDayReentryPolicy:
    """Policy controlling same-day reentry by exit type."""

    allow_on_rebalance: bool = True
    allow_on_max_holding: bool = False
    allow_on_stop_loss: bool = False
    allow_on_take_profit: bool = False
    allow_on_margin_liquidation: bool = False
    allow_on_partial_margin_liquidation: bool = False

    def allows(self, exit_type: str | None) -> bool:
        """Return True when same-day reentry is allowed for `exit_type`."""
        if exit_type is None:
            return False
        if exit_type == "Rebalance Period":
            return self.allow_on_rebalance
        if exit_type == "Max Holding Period":
            return self.allow_on_max_holding
        if exit_type == "Rebalance/Max Holding Period":
            return self.allow_on_rebalance or self.allow_on_max_holding
        if exit_type == "Stop Loss":
            return self.allow_on_stop_loss
        if exit_type == "Take Profit":
            return self.allow_on_take_profit
        if exit_type == "Margin Call Liquidation":
            return self.allow_on_margin_liquidation
        if exit_type == "Margin Call Partial Liquidation":
            return self.allow_on_partial_margin_liquidation
        if "/" in exit_type:
            return any(self.allows(part.strip()) for part in exit_type.split("/"))
        return False

    def allow_from_trade_rows(self, trade_rows: list[dict]) -> bool:
        """Evaluate same-day reentry from emitted trade rows."""
        for row in trade_rows:
            if self.allows(row.get("exit_type")):
                return True
        return False

    def allow_from_trade_records(self, trade_records: list[TradeRecord]) -> bool:
        """Evaluate same-day reentry from typed trade records."""
        for record in trade_records:
            if self.allows(record.exit_type):
                return True
        return False
