"""Registry helpers for config-driven signal and strategy preset resolution."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any, TypeAlias

from volatility_trading.signals import (
    LongOnlySignal,
    ShortOnlySignal,
    ZScoreSignal,
)
from volatility_trading.signals.base_signal import Signal
from volatility_trading.strategies.skew_mispricing import (
    SkewMispricingSpec,
    make_skew_mispricing_strategy,
)
from volatility_trading.strategies.vrp_harvesting import (
    VRPHarvestingSpec,
    make_vrp_strategy,
)

from ..options_engine.specs import StrategySpec
from .types import NamedSignalSpec, NamedStrategyPresetSpec

SignalFactory: TypeAlias = Callable[..., Signal]
StrategyPresetBuilder: TypeAlias = Callable[[NamedStrategyPresetSpec], StrategySpec]

_SIGNAL_FACTORIES: dict[str, SignalFactory] = {
    "long_only": LongOnlySignal,
    "short_only": ShortOnlySignal,
    "zscore": ZScoreSignal,
}
_VRP_ALLOWED_PARAMS = tuple(
    field.name
    for field in fields(VRPHarvestingSpec)
    if field.init and field.name != "signal"
)
_SKEW_ALLOWED_PARAMS = tuple(
    field.name
    for field in fields(SkewMispricingSpec)
    if field.init and field.name != "signal"
)


def _accepted_params_text(factory: Callable[..., Any]) -> str:
    """Describe accepted keyword params for a callable factory."""
    accepted = tuple(inspect.signature(factory).parameters)
    return ", ".join(accepted) if accepted else "none"


def _validate_factory_params(
    factory: Callable[..., Any],
    params: Mapping[str, Any],
    *,
    component_label: str,
    component_name: str,
) -> None:
    """Validate kwargs against a callable signature with a clear error."""
    try:
        inspect.signature(factory).bind(**params)
    except TypeError as exc:
        accepted = _accepted_params_text(factory)
        raise ValueError(
            f"Invalid parameters for {component_label} '{component_name}': {exc}. "
            f"Accepted parameters: {accepted}."
        ) from exc


def _unknown_name_error(
    component_label: str,
    component_name: str,
    available_names: tuple[str, ...],
) -> ValueError:
    """Build a deterministic unknown-name error message."""
    joined = ", ".join(available_names)
    return ValueError(
        f"Unknown {component_label} name '{component_name}'. "
        f"Available {component_label}s: {joined}."
    )


def _build_vrp_harvesting(spec: NamedStrategyPresetSpec) -> StrategySpec:
    """Build the current VRP harvesting preset from a named strategy spec."""
    return _build_dataclass_strategy_preset(
        spec,
        preset_factory=VRPHarvestingSpec,
        strategy_factory=make_vrp_strategy,
        allowed_params=_VRP_ALLOWED_PARAMS,
        allow_default_signal=False,
    )


def _build_skew_mispricing(spec: NamedStrategyPresetSpec) -> StrategySpec:
    """Build the skew mispricing preset from a named strategy spec."""
    return _build_dataclass_strategy_preset(
        spec,
        preset_factory=SkewMispricingSpec,
        strategy_factory=make_skew_mispricing_strategy,
        allowed_params=_SKEW_ALLOWED_PARAMS,
        allow_default_signal=True,
    )


def _build_dataclass_strategy_preset(
    spec: NamedStrategyPresetSpec,
    *,
    preset_factory: Callable[..., Any],
    strategy_factory: Callable[[Any], StrategySpec],
    allowed_params: tuple[str, ...],
    allow_default_signal: bool,
) -> StrategySpec:
    """Build one dataclass-backed strategy preset with shared validation."""
    unexpected = tuple(sorted(set(spec.params) - set(allowed_params)))
    if unexpected:
        accepted = ", ".join(allowed_params)
        unexpected_text = ", ".join(unexpected)
        raise ValueError(
            "Invalid parameters for strategy preset "
            f"'{spec.name}': unexpected parameters {unexpected_text}. "
            f"Accepted parameters: {accepted}."
        )

    params = dict(spec.params)
    if spec.signal is not None:
        params["signal"] = build_signal(spec.signal)
    elif not allow_default_signal:
        raise ValueError(
            f"Strategy preset '{spec.name}' requires a signal configuration."
        )

    preset = preset_factory(**params)
    return strategy_factory(preset)


_STRATEGY_PRESET_BUILDERS: dict[str, StrategyPresetBuilder] = {
    "skew_mispricing": _build_skew_mispricing,
    "vrp_harvesting": _build_vrp_harvesting,
}


def available_signal_names() -> tuple[str, ...]:
    """Return stable sorted signal names supported by the runner registry."""
    return tuple(sorted(_SIGNAL_FACTORIES))


def available_strategy_preset_names() -> tuple[str, ...]:
    """Return stable sorted strategy preset names supported by the runner."""
    return tuple(sorted(_STRATEGY_PRESET_BUILDERS))


def build_signal(spec: NamedSignalSpec) -> Signal:
    """Build a concrete signal instance from a named signal spec.

    Args:
        spec: Named signal config containing registry key and keyword params.

    Returns:
        Instantiated signal object suitable for strategy presets.

    Raises:
        ValueError: If the signal name is unknown or params do not match the
            registered signal constructor.
    """
    factory = _SIGNAL_FACTORIES.get(spec.name)
    if factory is None:
        raise _unknown_name_error("signal", spec.name, available_signal_names())

    _validate_factory_params(
        factory,
        spec.params,
        component_label="signal",
        component_name=spec.name,
    )
    return factory(**dict(spec.params))


def build_strategy_preset(spec: NamedStrategyPresetSpec) -> StrategySpec:
    """Build a concrete strategy spec from a named preset and nested signal.

    Args:
        spec: Named strategy preset config with strategy params and signal spec.

    Returns:
        Concrete `StrategySpec` ready for the existing backtest engine.

    Raises:
        ValueError: If the preset name is unknown or preset params are invalid.
    """
    builder = _STRATEGY_PRESET_BUILDERS.get(spec.name)
    if builder is None:
        raise _unknown_name_error(
            "strategy preset",
            spec.name,
            available_strategy_preset_names(),
        )
    return builder(spec)
