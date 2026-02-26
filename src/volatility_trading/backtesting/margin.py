"""Account-level margin lifecycle primitives for backtesting.

This module models path-dependent account behavior and intentionally lives in
`backtesting/` rather than `options/`.

Separation of concerns:
- `options/risk/margin.py`: position-level initial margin formulas (Reg-T/PM proxy)
- `backtesting/margin.py`: daily account lifecycle (maintenance, calls, liquidation,
  financing carry)

Lifecycle used by :class:`MarginAccount`:
1) consume an initial margin requirement for current open contracts
2) derive maintenance requirement (explicit per-contract input or policy ratio)
3) flag deficit when equity drops below maintenance requirement
4) count consecutive call days and trigger liquidation after grace period
5) compute optional financing carry for idle cash or borrowed balance
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Literal

import pandas as pd

from .rates import RateInput, coerce_rate_model
from .types import MarginCore


@dataclass(frozen=True)
class MarginPolicy:
    """Rules governing maintenance checks, calls, liquidation, and financing.

    Attributes:
        maintenance_margin_ratio: Fraction of initial margin used as maintenance
            when explicit maintenance input is not provided.
        margin_call_grace_days: Number of consecutive deficit days tolerated before
            forced liquidation can trigger.
        liquidation_mode: Liquidation behavior once grace is exceeded.
            `"full"` closes all open contracts. `"target"` only reduces contracts to
            a compliant level implied by equity.
        liquidation_buffer_ratio: Extra cushion used in target liquidation sizing.
            `0.10` means liquidate toward a 10% maintenance buffer.
        apply_financing: If True, apply daily financing carry on free cash and
            borrowed balances.
        cash_rate_annual: Financing rate input for free cash (constant, series,
            or model). Used only when `apply_financing=True`.
        borrow_rate_annual: Financing rate input for borrowed balance (constant,
            series, or model). Used only when `apply_financing=True`.
        trading_days_per_year: Day-count basis used to convert annual rates to
            daily carry.
    """

    maintenance_margin_ratio: float = 0.75
    margin_call_grace_days: int = 1
    liquidation_mode: Literal["full", "target"] = "full"
    liquidation_buffer_ratio: float = 0.0
    apply_financing: bool = False
    cash_rate_annual: RateInput = 0.0
    borrow_rate_annual: RateInput = 0.0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if not 0 <= self.maintenance_margin_ratio <= 1:
            raise ValueError("maintenance_margin_ratio must be in [0, 1]")
        if self.margin_call_grace_days < 0:
            raise ValueError("margin_call_grace_days must be >= 0")
        if self.liquidation_mode not in {"full", "target"}:
            raise ValueError("liquidation_mode must be 'full' or 'target'")
        if self.liquidation_buffer_ratio < 0:
            raise ValueError("liquidation_buffer_ratio must be >= 0")
        # Validate rate inputs early so misconfigured policies fail fast.
        coerce_rate_model(self.cash_rate_annual)
        coerce_rate_model(self.borrow_rate_annual)
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be > 0")


@dataclass(frozen=True)
class MarginStatus:
    """One-day account margin snapshot returned by :class:`MarginAccount`.

    Attributes:
        initial_margin_requirement: Initial margin requirement for current open
            contracts.
        core: Shared maintenance/call/liquidation/financing account state.
        contracts_to_liquidate: Number of contracts to close immediately.
        contracts_after_liquidation: Remaining open contracts after forced action.
        cash_balance: Positive idle cash used for financing accrual.
        borrowed_balance: Positive borrowed amount used for financing accrual.
    """

    initial_margin_requirement: float
    core: MarginCore
    contracts_to_liquidate: int
    contracts_after_liquidation: int
    cash_balance: float
    borrowed_balance: float


class MarginAccount:
    """Stateful evaluator for margin calls and liquidation decisions.

    The object is stateful because margin-call logic depends on consecutive
    breach days (`margin_call_days`), not only on today's inputs.
    """

    def __init__(self, policy: MarginPolicy):
        self.policy = policy
        self._margin_call_days = 0
        self._cash_rate_model = coerce_rate_model(policy.cash_rate_annual)
        self._borrow_rate_model = coerce_rate_model(policy.borrow_rate_annual)

    @property
    def margin_call_days(self) -> int:
        return self._margin_call_days

    def evaluate(
        self,
        *,
        equity: float,
        initial_margin_requirement: float,
        open_contracts: int,
        maintenance_margin_per_contract: float | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> MarginStatus:
        """Evaluate one daily account-margin step.

        Args:
            equity: Current account equity after market PnL for the step.
            initial_margin_requirement: Current initial margin requirement for all
                open contracts.
            open_contracts: Number of open contracts before any forced liquidation.
            maintenance_margin_per_contract: Optional explicit maintenance amount per
                contract. If omitted, this is derived from policy.
            as_of: Current valuation date used for date-dependent financing rates.

        Returns:
            MarginStatus with maintenance/call state, liquidation decision, and
            optional financing carry.
        """
        if open_contracts < 0:
            raise ValueError("open_contracts must be >= 0")

        initial_requirement = max(float(initial_margin_requirement), 0.0)
        equity_value = float(equity)
        maintenance_per_contract = self._maintenance_per_contract(
            initial_requirement=initial_requirement,
            open_contracts=open_contracts,
            maintenance_margin_per_contract=maintenance_margin_per_contract,
        )
        maintenance_requirement = maintenance_per_contract * open_contracts
        margin_excess = equity_value - maintenance_requirement
        margin_deficit = max(-margin_excess, 0.0)
        in_margin_call = open_contracts > 0 and margin_deficit > 0.0

        # Margin-call state is path-dependent: only consecutive breach days count.
        if in_margin_call:
            self._margin_call_days += 1
        else:
            self._margin_call_days = 0

        # Liquidation triggers only after grace window is exceeded.
        forced_liquidation = in_margin_call and (
            self._margin_call_days > self.policy.margin_call_grace_days
        )
        contracts_after = open_contracts
        contracts_to_liquidate = 0
        if forced_liquidation:
            contracts_after = self._contracts_after_liquidation(
                equity=equity_value,
                maintenance_margin_per_contract=maintenance_per_contract,
                open_contracts=open_contracts,
            )
            contracts_to_liquidate = max(open_contracts - contracts_after, 0)

        cash_balance, borrowed_balance, financing_pnl = self._financing_terms(
            equity=equity_value,
            initial_margin_requirement=initial_requirement,
            open_contracts=open_contracts,
            as_of=as_of,
        )
        core = MarginCore(
            financing_pnl=financing_pnl,
            maintenance_margin_requirement=maintenance_requirement,
            margin_excess=margin_excess,
            margin_deficit=margin_deficit,
            in_margin_call=in_margin_call,
            margin_call_days=self._margin_call_days,
            forced_liquidation=forced_liquidation,
            contracts_liquidated=contracts_to_liquidate,
        )

        return MarginStatus(
            initial_margin_requirement=initial_requirement,
            core=core,
            contracts_to_liquidate=contracts_to_liquidate,
            contracts_after_liquidation=contracts_after,
            cash_balance=cash_balance,
            borrowed_balance=borrowed_balance,
        )

    def _maintenance_per_contract(
        self,
        *,
        initial_requirement: float,
        open_contracts: int,
        maintenance_margin_per_contract: float | None,
    ) -> float:
        """Return maintenance margin per contract for the current step."""
        if open_contracts <= 0:
            return 0.0
        if maintenance_margin_per_contract is not None:
            return max(float(maintenance_margin_per_contract), 0.0)
        if initial_requirement <= 0:
            return 0.0
        return (
            initial_requirement / open_contracts
        ) * self.policy.maintenance_margin_ratio

    def _contracts_after_liquidation(
        self,
        *,
        equity: float,
        maintenance_margin_per_contract: float,
        open_contracts: int,
    ) -> int:
        """Return remaining contracts after forced liquidation.

        - `"full"` mode closes everything.
        - `"target"` mode keeps as many contracts as equity can support after
          applying the liquidation buffer.
        """
        if open_contracts <= 0:
            return 0
        if self.policy.liquidation_mode == "full":
            return 0

        per_contract = max(float(maintenance_margin_per_contract), 0.0)
        if per_contract <= 0:
            return 0

        denominator = per_contract * (1.0 + self.policy.liquidation_buffer_ratio)
        target = floor(max(float(equity), 0.0) / denominator)
        return max(min(target, open_contracts), 0)

    def _financing_terms(
        self,
        *,
        equity: float,
        initial_margin_requirement: float,
        open_contracts: int,
        as_of: pd.Timestamp | None,
    ) -> tuple[float, float, float]:
        """Compute financing balances and daily carry contribution.

        Positive carry comes from free cash. Negative carry comes from borrow.
        """
        if not self.policy.apply_financing or open_contracts <= 0:
            return 0.0, 0.0, 0.0

        cash_balance = max(equity - initial_margin_requirement, 0.0)
        borrowed_balance = max(initial_margin_requirement - equity, 0.0)
        day_count = float(self.policy.trading_days_per_year)
        cash_rate_annual = self._cash_rate_model.annual_rate(as_of=as_of)
        borrow_rate_annual = self._borrow_rate_model.annual_rate(as_of=as_of)
        financing_pnl = (
            cash_balance * cash_rate_annual / day_count
            - borrowed_balance * borrow_rate_annual / day_count
        )
        return cash_balance, borrowed_balance, financing_pnl
