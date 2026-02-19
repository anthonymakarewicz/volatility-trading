"""Constants for backtest reporting artifacts."""

from __future__ import annotations

from volatility_trading.config.paths import BACKTEST_REPORTS_ROOT

REPORT_VERSION = "1.0.0"

DEFAULT_REPORT_ROOT = BACKTEST_REPORTS_ROOT
EXPERIMENT_REPORT_ROOT = DEFAULT_REPORT_ROOT / "experiments"
OOS_REPORT_ROOT = DEFAULT_REPORT_ROOT / "oos"

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
