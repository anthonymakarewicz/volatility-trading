"""Position-sizing utilities for stress-based option risk budgets."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


def contracts_for_risk_budget(
    *,
    equity: float,
    risk_budget_pct: float,
    risk_per_contract: float,
    min_contracts: int = 0,
    max_contracts: int | None = None,
) -> int:
    """Convert a risk budget into an integer number of contracts.

    Sizing rule:
    `contracts = floor((equity * risk_budget_pct) / risk_per_contract)`.

    Notes:
        - If `risk_per_contract <= 0`, returns `min_contracts`.
        - `max_contracts` is applied after floor/min logic.
    """
    if equity < 0:
        raise ValueError("equity must be non-negative")
    if not 0 <= risk_budget_pct <= 1:
        raise ValueError("risk_budget_pct must be in [0, 1]")
    if min_contracts < 0:
        raise ValueError("min_contracts must be >= 0")
    if max_contracts is not None and max_contracts < min_contracts:
        raise ValueError("max_contracts must be >= min_contracts")

    if risk_per_contract <= 0:
        return min_contracts

    risk_budget = equity * risk_budget_pct
    contracts = max(floor(risk_budget / risk_per_contract), min_contracts)
    if max_contracts is not None:
        contracts = min(contracts, max_contracts)
    return int(contracts)


@dataclass(frozen=True)
class RiskBudgetSizer:
    """Configured wrapper around risk-budget contract sizing."""

    risk_budget_pct: float
    min_contracts: int = 0
    max_contracts: int | None = None

    def size(self, *, equity: float, risk_per_contract: float) -> int:
        """Return contract count allowed by the configured risk budget."""
        return contracts_for_risk_budget(
            equity=equity,
            risk_budget_pct=self.risk_budget_pct,
            risk_per_contract=risk_per_contract,
            min_contracts=self.min_contracts,
            max_contracts=self.max_contracts,
        )
