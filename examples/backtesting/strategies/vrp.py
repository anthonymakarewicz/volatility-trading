"""VRP strategy builder used by example scripts."""

from __future__ import annotations

from volatility_trading.backtesting import DeltaHedgePolicy
from volatility_trading.backtesting.options_engine import StrategySpec
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


def build_vrp_strategy(
    *,
    rebalance_period: int = 10,
    risk_budget_pct: float = 1.0,
    margin_budget_pct: float = 0.4,
    delta_hedge: DeltaHedgePolicy | None = None,
) -> StrategySpec:
    """Build one baseline VRP strategy used across examples."""
    return make_vrp_strategy(
        VRPHarvestingSpec(
            signal=ShortOnlySignal(),
            rebalance_period=rebalance_period,
            risk_budget_pct=risk_budget_pct,
            margin_budget_pct=margin_budget_pct,
            delta_hedge=delta_hedge or DeltaHedgePolicy(),
        )
    )
