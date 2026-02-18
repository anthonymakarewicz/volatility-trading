"""Constants for backtest reporting artifacts."""

from __future__ import annotations

from pathlib import Path

REPORT_VERSION = "1.0.0"
DEFAULT_REPORT_ROOT = Path("reports/backtests")

RUN_CONFIG_FILENAME = "run_config.json"
SUMMARY_METRICS_FILENAME = "summary_metrics.json"
EQUITY_DRAWDOWN_FILENAME = "equity_and_drawdown.csv"
TRADES_FILENAME = "trades.csv"
EXPOSURES_FILENAME = "exposures_daily.csv"
MANIFEST_FILENAME = "manifest.json"

PLOTS_DIRNAME = "plots"
DASHBOARD_FILENAME = "performance_dashboard.png"
EQUITY_FILENAME = "equity_vs_benchmark.png"
DRAWDOWN_FILENAME = "drawdown.png"
GREEKS_FILENAME = "greeks_exposure.png"
