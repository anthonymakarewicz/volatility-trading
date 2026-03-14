"""Shared helpers for executable examples."""

from .backtesting_helpers import (
    build_backtester,
    build_data_bundle,
    build_run_config,
    load_daily_features_window,
    load_options_window,
    load_rf_series,
    run_and_report,
)
from .cli import (
    CommonExampleConfig,
    VrpExampleConfig,
    parse_common_args,
    parse_vrp_args,
)

__all__ = [
    "CommonExampleConfig",
    "VrpExampleConfig",
    "parse_common_args",
    "parse_vrp_args",
    "load_daily_features_window",
    "load_options_window",
    "load_rf_series",
    "build_run_config",
    "run_and_report",
    "build_data_bundle",
    "build_backtester",
]
