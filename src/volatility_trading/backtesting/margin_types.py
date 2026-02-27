"""Shared margin/accounting dataclasses used across backtesting modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarginCore:
    """Shared margin/accounting state used across backtesting lifecycle snapshots."""

    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int

    @classmethod
    def empty(cls) -> MarginCore:
        """Return default non-margin-call state used before account evaluation."""
        return cls(
            financing_pnl=0.0,
            maintenance_margin_requirement=0.0,
            margin_excess=float("nan"),
            margin_deficit=float("nan"),
            in_margin_call=False,
            margin_call_days=0,
            forced_liquidation=False,
            contracts_liquidated=0,
        )
