"""Constants for backtest reporting artifacts."""

from __future__ import annotations

from volatility_trading.config.paths import BACKTEST_REPORTS_ROOT

REPORT_VERSION = "1.4.0"
ROLLING_METRICS_WINDOW = 63

DEFAULT_REPORT_ROOT = BACKTEST_REPORTS_ROOT
EXPERIMENT_REPORT_ROOT = DEFAULT_REPORT_ROOT / "experiments"
OOS_REPORT_ROOT = DEFAULT_REPORT_ROOT / "oos"

RUN_CONFIG_FILENAME = "run_config.json"
SUMMARY_METRICS_FILENAME = "summary_metrics.json"
EQUITY_DRAWDOWN_FILENAME = "equity_and_drawdown.csv"
TRADES_FILENAME = "trades.csv"
ENTRY_STRESS_DIAGNOSTICS_FILENAME = "entry_stress_diagnostics.parquet"
STRESS_SCENARIO_SUMMARY_FILENAME = "stress_scenario_summary.csv"
EXPOSURES_FILENAME = "exposures_daily.csv"
MARGIN_DIAGNOSTICS_FILENAME = "margin_diagnostics_daily.csv"
ROLLING_METRICS_FILENAME = "rolling_metrics.csv"
PNL_ATTRIBUTION_FILENAME = "pnl_attribution_daily.csv"
BENCHMARK_COMPARISON_FILENAME = "benchmark_comparison.json"
MANIFEST_FILENAME = "manifest.json"

PLOTS_DIRNAME = "plots"
DASHBOARD_FILENAME = "performance_dashboard.png"
EQUITY_FILENAME = "equity_vs_benchmark.png"
DRAWDOWN_FILENAME = "drawdown.png"
GREEKS_FILENAME = "greeks_exposure.png"
MARGIN_ACCOUNT_FILENAME = "margin_account.png"
ROLLING_METRICS_PLOT_FILENAME = "rolling_metrics.png"
PNL_ATTRIBUTION_PLOT_FILENAME = "pnl_attribution.png"
