"""Typed workflow contracts for config-driven backtest runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from volatility_trading.backtesting.config import (
    AccountConfig,
    BacktestRunConfig,
    BrokerConfig,
    ExecutionConfig,
    ModelingConfig,
)
from volatility_trading.backtesting.reporting.constants import DEFAULT_REPORT_ROOT

from .catalog import (
    FEATURES_SOURCE_PROVIDERS,
    OPTIONS_SOURCE_PROVIDERS,
    RATES_SOURCE_PROVIDERS,
    SERIES_SOURCE_PROVIDERS,
)


def _normalize_name(value: str, *, field_name: str) -> str:
    """Normalize a registry component name and reject blanks."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _freeze_params(params: Mapping[str, Any]) -> MappingProxyType:
    """Freeze config params so runner specs behave like immutable config."""
    return MappingProxyType(dict(params))


def _normalize_required_name(value: str, *, field_name: str) -> str:
    """Normalize one required label and reject blank strings."""
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _normalize_optional_name(value: str | None, *, field_name: str) -> str | None:
    """Normalize an optional label and reject blank-but-present strings."""
    if value is None:
        return None
    return _normalize_required_name(value, field_name=field_name)


def _coerce_timestamp(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    """Coerce optional date-like values to pandas timestamps."""
    if value is None:
        return None
    return pd.Timestamp(value)


def _validate_provider(
    provider: str,
    *,
    field_name: str,
    allowed: tuple[str, ...],
) -> str:
    """Validate one provider name against an allowed list."""
    normalized = _normalize_required_name(provider, field_name=field_name).lower()
    if normalized not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(
            f"{field_name} must be one of {allowed_text}; got '{provider}'."
        )
    return normalized


def _validate_positive_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    """Validate a positive finite scalar."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and > 0")
    return float(value)


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
    signal: NamedSignalSpec | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_name(self.name, field_name="strategy preset name"),
        )
        object.__setattr__(self, "params", _freeze_params(self.params))


@dataclass(frozen=True)
class OptionsSourceSpec:
    """Typed options-chain source spec for one backtest workflow."""

    ticker: str
    provider: str = "orats"
    proc_root: Path | None = None
    adapter_name: str | None = None
    symbol: str | None = None
    default_contract_multiplier: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ticker",
            _normalize_required_name(self.ticker, field_name="options ticker").upper(),
        )
        object.__setattr__(
            self,
            "provider",
            _validate_provider(
                self.provider,
                field_name="options provider",
                allowed=OPTIONS_SOURCE_PROVIDERS,
            ),
        )
        object.__setattr__(
            self,
            "adapter_name",
            _normalize_optional_name(
                self.adapter_name,
                field_name="options adapter_name",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_optional_name(self.symbol, field_name="options symbol"),
        )
        object.__setattr__(
            self,
            "default_contract_multiplier",
            _validate_positive_finite(
                self.default_contract_multiplier,
                field_name="default_contract_multiplier",
            ),
        )


@dataclass(frozen=True)
class FeaturesSourceSpec:
    """Typed features-panel source spec for one backtest workflow."""

    ticker: str
    provider: str = "orats"
    proc_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ticker",
            _normalize_required_name(self.ticker, field_name="features ticker").upper(),
        )
        object.__setattr__(
            self,
            "provider",
            _validate_provider(
                self.provider,
                field_name="features provider",
                allowed=FEATURES_SOURCE_PROVIDERS,
            ),
        )


@dataclass(frozen=True)
class SeriesSourceSpec:
    """Typed time-series source spec for hedge or benchmark inputs."""

    ticker: str
    provider: str = "yfinance"
    proc_root: Path | None = None
    price_column: str = "close"
    symbol: str | None = None
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ticker",
            _normalize_required_name(self.ticker, field_name="series ticker").upper(),
        )
        object.__setattr__(
            self,
            "provider",
            _validate_provider(
                self.provider,
                field_name="series provider",
                allowed=SERIES_SOURCE_PROVIDERS,
            ),
        )
        object.__setattr__(
            self,
            "price_column",
            _normalize_required_name(
                self.price_column,
                field_name="series price_column",
            ),
        )
        object.__setattr__(
            self,
            "symbol",
            _normalize_optional_name(self.symbol, field_name="series symbol"),
        )
        object.__setattr__(
            self,
            "contract_multiplier",
            _validate_positive_finite(
                self.contract_multiplier,
                field_name="series contract_multiplier",
            ),
        )


@dataclass(frozen=True)
class RatesSourceSpec:
    """Typed rates source spec for financing and report inputs."""

    provider: str = "constant"
    constant_rate: float | None = 0.0
    series_id: str | None = None
    column: str | None = None
    proc_root: Path | None = None

    def __post_init__(self) -> None:
        provider = _validate_provider(
            self.provider,
            field_name="rates provider",
            allowed=RATES_SOURCE_PROVIDERS,
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "series_id",
            _normalize_optional_name(self.series_id, field_name="rates series_id"),
        )
        object.__setattr__(
            self,
            "column",
            _normalize_optional_name(self.column, field_name="rates column"),
        )

        if provider == "constant":
            if self.constant_rate is None or not math.isfinite(self.constant_rate):
                raise ValueError(
                    "constant-rate source requires constant_rate to be finite"
                )
        else:
            if self.series_id is None:
                raise ValueError("fred rates source requires a non-empty series_id")


@dataclass(frozen=True)
class BacktestDataSourcesSpec:
    """Typed data-source bundle spec for one config-driven backtest workflow."""

    options: OptionsSourceSpec
    features: FeaturesSourceSpec | None = None
    hedge: SeriesSourceSpec | None = None
    benchmark: SeriesSourceSpec | None = None
    rates: RatesSourceSpec | None = None


@dataclass(frozen=True)
class RunWindowSpec:
    """Typed date-window spec for one backtest workflow."""

    start_date: pd.Timestamp | str | None = None
    end_date: pd.Timestamp | str | None = None

    def __post_init__(self) -> None:
        start = _coerce_timestamp(self.start_date)
        end = _coerce_timestamp(self.end_date)
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        if start is not None and end is not None and start > end:
            raise ValueError("run window start_date must be <= end_date")


@dataclass(frozen=True)
class ReportingSpec:
    """Typed reporting-output spec for one config-driven backtest workflow."""

    output_root: Path = DEFAULT_REPORT_ROOT
    run_id: str | None = None
    benchmark_name: str | None = None
    include_dashboard_plot: bool = True
    include_component_plots: bool = False
    save_report_bundle: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _normalize_optional_name(self.run_id, field_name="reporting run_id"),
        )
        object.__setattr__(
            self,
            "benchmark_name",
            _normalize_optional_name(
                self.benchmark_name,
                field_name="reporting benchmark_name",
            ),
        )


@dataclass(frozen=True)
class BacktestWorkflowSpec:
    """Top-level typed workflow spec for future YAML/CLI backtest runs."""

    data: BacktestDataSourcesSpec
    strategy: NamedStrategyPresetSpec
    run: RunWindowSpec = field(default_factory=RunWindowSpec)
    reporting: ReportingSpec = field(default_factory=ReportingSpec)
    account: AccountConfig = field(default_factory=AccountConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)

    def to_backtest_run_config(self) -> BacktestRunConfig:
        """Project this workflow spec onto the existing engine run config."""
        return BacktestRunConfig(
            account=self.account,
            execution=self.execution,
            broker=self.broker,
            modeling=self.modeling,
            start_date=self.run.start_date,
            end_date=self.run.end_date,
        )
