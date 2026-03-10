"""Typed named-component specs for config-driven backtest workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _normalize_name(value: str, *, field_name: str) -> str:
    """Normalize a registry component name and reject blanks."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _freeze_params(params: Mapping[str, Any]) -> MappingProxyType:
    """Freeze config params so runner specs behave like immutable config."""
    return MappingProxyType(dict(params))


@dataclass(frozen=True)
class NamedSignalSpec:
    """Named signal config used by the future backtest runner."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_name(self.name, field_name="signal name"),
        )
        object.__setattr__(self, "params", _freeze_params(self.params))


@dataclass(frozen=True)
class NamedStrategyPresetSpec:
    """Named strategy preset config used by the future backtest runner."""

    name: str
    signal: NamedSignalSpec
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_name(self.name, field_name="strategy preset name"),
        )
        object.__setattr__(self, "params", _freeze_params(self.params))
