"""Skew-mispricing strategy builder used by example scripts."""

from __future__ import annotations

from volatility_trading.backtesting import DeltaHedgePolicy
from volatility_trading.backtesting.options_engine import StrategySpec
from volatility_trading.strategies import (
    SkewMispricingSpec,
    make_skew_mispricing_strategy,
)


def build_skew_strategy(
    *,
    target_dte: int = 30,
    max_holding_period: int = 30,
    risk_budget_pct: float = 1.0,
    margin_budget_pct: float = 0.4,
    delta_hedge: DeltaHedgePolicy | None = None,
) -> StrategySpec:
    """Build one baseline skew-mispricing strategy used across examples."""
    return make_skew_mispricing_strategy(
        SkewMispricingSpec(
            target_dte=target_dte,
            max_holding_period=max_holding_period,
            risk_budget_pct=risk_budget_pct,
            margin_budget_pct=margin_budget_pct,
            delta_hedge=delta_hedge or DeltaHedgePolicy(),
        )
    )
