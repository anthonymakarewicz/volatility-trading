"""High-level orchestration for building and saving backtest reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .builders import (
    build_equity_and_drawdown_table,
    build_exposures_daily_table,
    build_summary_metrics,
)
from .constants import (
    DASHBOARD_FILENAME,
    DEFAULT_REPORT_ROOT,
    DRAWDOWN_FILENAME,
    EQUITY_FILENAME,
    GREEKS_FILENAME,
)
from .plots import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_greeks_exposure,
    plot_performance_dashboard,
)
from .schemas import BacktestReportBundle, ReportMetadata
from .writers import write_report_bundle


def create_run_id(*, now: datetime | None = None) -> str:
    """Create a sortable UTC run id."""
    timestamp = now or datetime.now(tz=UTC)
    return timestamp.strftime("%Y%m%d_%H%M%S")


def build_backtest_report_bundle(
    *,
    trades: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    run_config: dict[str, Any],
    strategy_name: str,
    benchmark: pd.Series | None = None,
    benchmark_name: str | None = None,
    run_id: str | None = None,
    include_dashboard_plot: bool = True,
    include_component_plots: bool = False,
    risk_free_rate: float = 0.0,
) -> BacktestReportBundle:
    """Build a complete in-memory report bundle for one backtest run."""
    if mtm_daily.empty:
        raise ValueError("mtm_daily must not be empty")

    now = datetime.now(tz=UTC)
    resolved_run_id = run_id or create_run_id(now=now)
    metadata = ReportMetadata(
        strategy_name=strategy_name,
        run_id=resolved_run_id,
        created_at_utc=now.isoformat(),
        benchmark_name=benchmark_name,
    )

    summary = build_summary_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=risk_free_rate,
    )
    equity_and_drawdown = build_equity_and_drawdown_table(
        mtm_daily=mtm_daily, benchmark=benchmark
    )
    exposures = build_exposures_daily_table(mtm_daily=mtm_daily)

    figures = {}
    if include_dashboard_plot:
        figures[DASHBOARD_FILENAME] = plot_performance_dashboard(
            benchmark=benchmark,
            mtm_daily=mtm_daily,
            strategy_name=strategy_name,
            benchmark_name=benchmark_name or "Benchmark",
        )
    if include_component_plots:
        figures[EQUITY_FILENAME] = plot_equity_vs_benchmark(
            benchmark=benchmark,
            mtm_daily=mtm_daily,
            strategy_name=strategy_name,
            benchmark_name=benchmark_name or "Benchmark",
        )
        figures[DRAWDOWN_FILENAME] = plot_drawdown(
            benchmark=benchmark,
            mtm_daily=mtm_daily,
            benchmark_name=benchmark_name or "Benchmark",
        )
        figures[GREEKS_FILENAME] = plot_greeks_exposure(mtm_daily=mtm_daily)

    return BacktestReportBundle(
        metadata=metadata,
        run_config=run_config,
        summary_metrics=summary,
        equity_and_drawdown=equity_and_drawdown,
        trades=trades.copy(),
        exposures_daily=exposures,
        figures=figures,
    )


def save_backtest_report_bundle(
    bundle: BacktestReportBundle,
    *,
    output_root: Path = DEFAULT_REPORT_ROOT,
) -> Path:
    """Persist report bundle artifacts to disk and return the run directory."""
    return write_report_bundle(bundle, output_root=output_root)
