"""Public runtime output records for options backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.margin_types import MarginCore
from volatility_trading.options.types import Greeks, MarketState


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
class TradeLegRecord:
    """One per-leg payload row attached to a realized trade record."""

    leg_index: int
    option_type: str
    strike: float
    expiry_date: pd.Timestamp | None
    weight: int
    side: int
    effective_side: int
    entry_price: float
    exit_price: float
    delta_target: float
    delta_tolerance: float
    expiry_group: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the per-leg payload into the canonical mapping schema."""
        return {
            "leg_index": self.leg_index,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry_date": self.expiry_date,
            "weight": self.weight,
            "side": self.side,
            "effective_side": self.effective_side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "delta_target": self.delta_target,
            "delta_tolerance": self.delta_tolerance,
            "expiry_group": self.expiry_group,
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
    trade_legs: tuple[TradeLegRecord, ...]

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
            "trade_legs": [leg.to_dict() for leg in self.trade_legs],
        }
