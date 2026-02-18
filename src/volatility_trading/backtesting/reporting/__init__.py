"""Backtest reporting builders, plots, and writers."""

from .builders import (
    build_equity_and_drawdown_table,
    build_exposures_daily_table,
    build_summary_metrics,
)
from .plots import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_greeks_exposure,
    plot_performance_dashboard,
)
from .schemas import BacktestReportBundle, ReportMetadata, SummaryMetrics
from .service import (
    build_backtest_report_bundle,
    create_run_id,
    save_backtest_report_bundle,
)
from .writers import write_report_bundle

__all__ = [
    "ReportMetadata",
    "SummaryMetrics",
    "BacktestReportBundle",
    "create_run_id",
    "build_summary_metrics",
    "build_equity_and_drawdown_table",
    "build_exposures_daily_table",
    "plot_equity_vs_benchmark",
    "plot_drawdown",
    "plot_greeks_exposure",
    "plot_performance_dashboard",
    "build_backtest_report_bundle",
    "save_backtest_report_bundle",
    "write_report_bundle",
]
