"""Internal building blocks for config-driven backtest workflows."""

from .config_parser import parse_workflow_config
from .registry import (
    available_signal_names,
    available_strategy_preset_names,
    build_signal,
    build_strategy_preset,
)
from .types import NamedSignalSpec, NamedStrategyPresetSpec
from .workflow_types import (
    BacktestDataSourcesSpec,
    BacktestWorkflowSpec,
    FeaturesSourceSpec,
    OptionsSourceSpec,
    RatesSourceSpec,
    ReportingSpec,
    RunWindowSpec,
    SeriesSourceSpec,
)

__all__ = [
    "NamedSignalSpec",
    "NamedStrategyPresetSpec",
    "OptionsSourceSpec",
    "FeaturesSourceSpec",
    "SeriesSourceSpec",
    "RatesSourceSpec",
    "BacktestDataSourcesSpec",
    "RunWindowSpec",
    "ReportingSpec",
    "BacktestWorkflowSpec",
    "parse_workflow_config",
    "available_signal_names",
    "available_strategy_preset_names",
    "build_signal",
    "build_strategy_preset",
]
