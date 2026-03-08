"""Shared helpers for executable examples."""

from .cli import (
    CommonExampleConfig,
    VrpExampleConfig,
    parse_common_args,
    parse_vrp_args,
)
from .vrp_helpers import (
    build_backtester,
    build_data_bundle,
    build_run_config,
    load_options_long,
    load_rf_series,
    run_and_report,
)

__all__ = [
    "CommonExampleConfig",
    "VrpExampleConfig",
    "parse_common_args",
    "parse_vrp_args",
    "load_options_long",
    "load_rf_series",
    "build_data_bundle",
    "build_run_config",
    "build_backtester",
    "run_and_report",
]
