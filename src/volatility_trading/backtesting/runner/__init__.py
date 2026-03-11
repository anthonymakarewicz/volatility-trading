"""Internal building blocks for config-driven backtest workflows."""

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
    "available_signal_names",
    "available_strategy_preset_names",
    "build_signal",
    "build_strategy_preset",
]
