"""Run-level backtesting configuration contracts (account, execution, broker, modeling)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from volatility_trading.options import (
        MarginModel,
        PriceModel,
        RiskEstimator,
        ScenarioGenerator,
    )

    from .margin import MarginPolicy


def _default_pricer() -> PriceModel:
    """Create default option pricer for one backtest run config."""
    from volatility_trading.options import BlackScholesPricer

    return BlackScholesPricer()


def _default_scenario_generator() -> ScenarioGenerator:
    """Create default stress-scenario generator for one backtest run config."""
    from volatility_trading.options import FixedGridScenarioGenerator

    return FixedGridScenarioGenerator()


def _default_risk_estimator() -> RiskEstimator:
    """Create default risk estimator for one backtest run config."""
    from volatility_trading.options import StressLossRiskEstimator

    return StressLossRiskEstimator()


@dataclass(frozen=True)
class AccountConfig:
    """Account-level capital configuration for one backtest run."""

    initial_capital: float = 100_000.0


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions used by entry/exit lifecycle pricing."""

    lot_size: int = 100
    slip_ask: float = 0.01
    slip_bid: float = 0.01
    commission_per_leg: float = 1.0

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if self.slip_ask < 0:
            raise ValueError("slip_ask must be >= 0")
        if self.slip_bid < 0:
            raise ValueError("slip_bid must be >= 0")
        if self.commission_per_leg < 0:
            raise ValueError("commission_per_leg must be >= 0")


@dataclass(frozen=True)
class MarginConfig:
    """Broker margin primitives used by sizing and margin-account lifecycle."""

    model: MarginModel | None = None
    policy: MarginPolicy | None = None


@dataclass(frozen=True)
class BrokerConfig:
    """Execution venue / broker rules for one backtest run."""

    margin: MarginConfig = field(default_factory=MarginConfig)


@dataclass(frozen=True)
class ModelingConfig:
    """Runtime pricing/risk engines used by sizing and lifecycle valuation."""

    pricer: PriceModel = field(default_factory=_default_pricer)
    scenario_generator: ScenarioGenerator = field(
        default_factory=_default_scenario_generator
    )
    risk_estimator: RiskEstimator = field(default_factory=_default_risk_estimator)


@dataclass(frozen=True)
class BacktestRunConfig:
    """Run-level backtesting configuration for one simulation."""

    account: AccountConfig = field(default_factory=AccountConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.start_date is not None and self.end_date is not None:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must be <= end_date")
