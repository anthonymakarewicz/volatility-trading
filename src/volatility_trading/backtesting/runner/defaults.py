"""Shared internal defaults for the backtest runner CLI/workflow."""

from __future__ import annotations

from typing import Any

from volatility_trading.backtesting.reporting.constants import DEFAULT_REPORT_ROOT
from volatility_trading.cli import DEFAULT_LOGGING

RUNNER_APP_ONLY_CONFIG_KEYS = frozenset({"dry_run", "logging"})

DEFAULT_RUNNER_APP_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "data": {
        "options": {
            "ticker": "SPY",
            "provider": "orats",
            "proc_root": None,
            "adapter_name": None,
            "symbol": None,
            "default_contract_multiplier": 100.0,
            "dte_min": None,
            "dte_max": None,
        },
        "rates": {
            "provider": "constant",
            "constant_rate": 0.0,
            "series_id": None,
            "column": None,
            "proc_root": None,
        },
    },
    "account": {
        "initial_capital": 100_000.0,
    },
    "execution": {
        "option": {
            "model": "bid_ask_fee",
            "params": {},
        },
        "hedge": {
            "model": "fixed_bps",
            "params": {},
        },
    },
    "broker": {},
    "modeling": {},
    "run": {
        "start_date": None,
        "end_date": None,
    },
    "reporting": {
        "output_root": DEFAULT_REPORT_ROOT,
        "run_id": None,
        "benchmark_name": None,
        "include_dashboard_plot": True,
        "include_component_plots": False,
        "save_report_bundle": True,
    },
}
