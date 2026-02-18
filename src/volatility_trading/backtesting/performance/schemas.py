"""Dataclasses for computed backtest performance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReturnMetrics:
    """Return and volatility statistics."""

    total_return: float | None
    cagr: float | None
    annualized_volatility: float | None
    sharpe: float | None


@dataclass(frozen=True)
class DrawdownMetrics:
    """Drawdown depth and duration statistics."""

    max_drawdown: float | None
    average_drawdown: float | None
    max_drawdown_duration_days: int


@dataclass(frozen=True)
class TailMetrics:
    """Tail-risk statistics from daily return distribution."""

    alpha: float
    var: float | None
    cvar: float | None


@dataclass(frozen=True)
class TradeMetrics:
    """Trade-level and realized PnL summary statistics."""

    total_trades: int
    win_rate: float | None
    average_win_pnl: float | None
    average_loss_pnl: float | None
    total_pnl: float | None
    gross_gain: float | None
    gross_loss: float | None
    profit_factor: float | None
    trade_frequency_per_year: float | None


@dataclass(frozen=True)
class PerformanceMetricsBundle:
    """Container for all computed metrics used by console/reporting layers."""

    returns: ReturnMetrics
    drawdown: DrawdownMetrics
    tail: TailMetrics
    trades: TradeMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return nested dict representation for serialization/inspection."""
        return asdict(self)
