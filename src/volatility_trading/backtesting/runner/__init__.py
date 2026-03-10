"""Internal building blocks for config-driven backtest workflows."""

from .registry import (
    available_signal_names,
    available_strategy_preset_names,
    build_signal,
    build_strategy_preset,
)
from .types import NamedSignalSpec, NamedStrategyPresetSpec

__all__ = [
    "NamedSignalSpec",
    "NamedStrategyPresetSpec",
    "available_signal_names",
    "available_strategy_preset_names",
    "build_signal",
    "build_strategy_preset",
]
