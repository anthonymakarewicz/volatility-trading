"""Exit-rule framework and same-day reentry policy for options strategies.

This module isolates lifecycle close triggers from execution mechanics so
strategies can compose exit behavior without rewriting lifecycle code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from ._lifecycle.ledger import TradeRecord

# TODO: Add Stop Loss and TP Exit (maybe for liquidated positon because of unsufficient margin balance ? )


@runtime_checkable
class HasPeriodicExitDates(Protocol):
    """Minimal position contract required by periodic exit rules."""

    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None


@runtime_checkable
class ExitRule(Protocol):
    """Contract for one exit rule evaluated against an open position."""

    exit_type: str

    def is_triggered(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> bool:
        """Return True when the rule requires position close on `curr_date`."""
        ...


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
class ExitRuleSet:
    """Set of exit rules evaluated together for one position/date."""

    rules: tuple[ExitRule, ...]
    combined_rebalance_max_hold_type: str = "Rebalance/Max Holding Period"

    @classmethod
    def period_rules(cls) -> ExitRuleSet:
        """Return default periodic exit rules (rebalance + max-holding)."""
        return cls(rules=(RebalanceExitRule(), MaxHoldingExitRule()))

    def evaluate(
        self,
        *,
        curr_date: pd.Timestamp,
        position: HasPeriodicExitDates,
    ) -> str | None:
        """Return exit-type label when one or more rules are triggered."""
        triggered = tuple(
            rule.exit_type
            for rule in self.rules
            if rule.is_triggered(curr_date=curr_date, position=position)
        )
        if not triggered:
            return None
        if len(triggered) == 1:
            return triggered[0]
        if {"Rebalance Period", "Max Holding Period"}.issubset(set(triggered)):
            return self.combined_rebalance_max_hold_type
        return "/".join(triggered)


@dataclass(frozen=True)
class SameDayReentryPolicy:
    """Policy controlling same-day reentry by exit type."""

    allow_on_rebalance: bool = True
    allow_on_max_holding: bool = False
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
        if exit_type == "Margin Call Liquidation":
            return self.allow_on_margin_liquidation
        if exit_type == "Margin Call Partial Liquidation":
            return self.allow_on_partial_margin_liquidation
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
