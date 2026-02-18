"""Dataclasses for backtest reporting bundle construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from .constants import REPORT_VERSION


@dataclass(frozen=True)
class ReportMetadata:
    """Identity and reproducibility metadata for one report run."""

    strategy_name: str
    run_id: str
    created_at_utc: str
    report_version: str = REPORT_VERSION
    benchmark_name: str | None = None
    git_commit: str | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    """Compact headline performance metrics for one run."""

    total_return: float | None
    cagr: float | None
    annualized_volatility: float | None
    sharpe: float | None
    max_drawdown: float | None
    total_trades: int
    win_rate: float | None
    profit_factor: float | None


@dataclass
class BacktestReportBundle:
    """In-memory report artifacts before filesystem persistence."""

    metadata: ReportMetadata
    run_config: dict[str, Any]
    summary_metrics: SummaryMetrics
    equity_and_drawdown: pd.DataFrame
    trades: pd.DataFrame
    exposures_daily: pd.DataFrame
    figures: dict[str, Figure] = field(default_factory=dict)
