"""Public API for high-level backtest reporting services."""

from .constants import (
    DEFAULT_REPORT_ROOT,
    EXPERIMENT_REPORT_ROOT,
    OOS_REPORT_ROOT,
)
from .plots import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_greeks_exposure,
    plot_performance_dashboard,
    plot_pnl_attribution,
    plot_stressed_pnl,
)
from .schemas import BacktestReportBundle, ReportMetadata, SummaryMetrics
from .service import (
    build_backtest_report_bundle,
    create_run_id,
    save_backtest_report_bundle,
)

__all__ = [
    "ReportMetadata",
    "SummaryMetrics",
    "BacktestReportBundle",
    "DEFAULT_REPORT_ROOT",
    "EXPERIMENT_REPORT_ROOT",
    "OOS_REPORT_ROOT",
    "create_run_id",
    "build_backtest_report_bundle",
    "save_backtest_report_bundle",
    "plot_equity_vs_benchmark",
    "plot_drawdown",
    "plot_greeks_exposure",
    "plot_pnl_attribution",
    "plot_stressed_pnl",
    "plot_performance_dashboard",
]
