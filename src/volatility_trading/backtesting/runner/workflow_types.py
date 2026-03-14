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
    MarginConfig,
    ModelingConfig,
)
from volatility_trading.backtesting.margin import MarginPolicy
from volatility_trading.backtesting.rates import RateInput
from volatility_trading.backtesting.reporting.constants import DEFAULT_REPORT_ROOT

from .source_loaders import (
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


def _validate_non_negative_finite(
    value: float,
    *,
    field_name: str,
) -> float:
    """Validate a non-negative finite scalar."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and >= 0")
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
    dte_min: float | None = None
    dte_max: float | None = None

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
        if self.dte_min is not None:
            object.__setattr__(
                self,
                "dte_min",
                _validate_non_negative_finite(
                    self.dte_min,
                    field_name="options dte_min",
                ),
            )
        if self.dte_max is not None:
            object.__setattr__(
                self,
                "dte_max",
                _validate_non_negative_finite(
                    self.dte_max,
                    field_name="options dte_max",
                ),
            )
        if (
            self.dte_min is not None
            and self.dte_max is not None
            and self.dte_min > self.dte_max
        ):
            raise ValueError("options dte_min must be <= dte_max")


def _add_rate_spread(
    rate_input: RateInput,
    *,
    spread: float,
) -> RateInput:
    """Apply one additive annualized spread to a runner rate input."""
    spread_value = _validate_non_negative_finite(
        spread,
        field_name="borrow_rate_spread",
    )
    if isinstance(rate_input, pd.Series):
        return pd.Series(
            pd.to_numeric(rate_input, errors="coerce").astype(float).values
            + spread_value,
            index=pd.DatetimeIndex(rate_input.index),
            name=rate_input.name,
        )
    return float(rate_input) + spread_value


@dataclass(frozen=True)
class MarginPolicySpec:
    """Runner-only margin-policy spec that can resolve data-sourced financing."""

    maintenance_margin_ratio: float = 0.75
    margin_call_grace_days: int = 1
    liquidation_mode: str = "full"
    liquidation_buffer_ratio: float = 0.0
    apply_financing: bool = False
    cash_rate_annual: float | None = None
    cash_rate_source: str | None = None
    borrow_rate_annual: float | None = None
    borrow_rate_spread: float | None = None
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cash_rate_source",
            _normalize_optional_name(
                self.cash_rate_source,
                field_name="margin policy cash_rate_source",
            ),
        )
        if not 0 <= self.maintenance_margin_ratio <= 1:
            raise ValueError("maintenance_margin_ratio must be in [0, 1]")
        if self.margin_call_grace_days < 0:
            raise ValueError("margin_call_grace_days must be >= 0")
        if self.liquidation_mode not in {"full", "target"}:
            raise ValueError("liquidation_mode must be 'full' or 'target'")
        object.__setattr__(
            self,
            "liquidation_buffer_ratio",
            _validate_non_negative_finite(
                self.liquidation_buffer_ratio,
                field_name="liquidation_buffer_ratio",
            ),
        )
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be > 0")
        if self.cash_rate_source is not None and self.cash_rate_source != "data_rates":
            raise ValueError("cash_rate_source must be 'data_rates' when provided")
        if self.cash_rate_source is not None and self.cash_rate_annual is not None:
            raise ValueError(
                "cash_rate_annual and cash_rate_source are mutually exclusive"
            )
        if self.borrow_rate_annual is not None and self.borrow_rate_spread is not None:
            raise ValueError(
                "borrow_rate_annual and borrow_rate_spread are mutually exclusive"
            )
        if (
            self.borrow_rate_spread is not None
            and self.cash_rate_source != "data_rates"
        ):
            raise ValueError(
                "borrow_rate_spread requires cash_rate_source='data_rates'"
            )
        if self.cash_rate_annual is not None:
            object.__setattr__(
                self,
                "cash_rate_annual",
                _validate_non_negative_finite(
                    self.cash_rate_annual,
                    field_name="cash_rate_annual",
                ),
            )
        if self.borrow_rate_annual is not None:
            object.__setattr__(
                self,
                "borrow_rate_annual",
                _validate_non_negative_finite(
                    self.borrow_rate_annual,
                    field_name="borrow_rate_annual",
                ),
            )
        if self.borrow_rate_spread is not None:
            object.__setattr__(
                self,
                "borrow_rate_spread",
                _validate_non_negative_finite(
                    self.borrow_rate_spread,
                    field_name="borrow_rate_spread",
                ),
            )

    def resolve(
        self,
        *,
        data_rates: RateInput | None,
    ) -> MarginPolicy:
        """Resolve one runner policy spec into the runtime MarginPolicy."""
        if self.cash_rate_source == "data_rates":
            if data_rates is None:
                raise ValueError(
                    "cash_rate_source='data_rates' requires data.rates in the workflow"
                )
            cash_rate: RateInput = data_rates
        else:
            cash_rate = (
                0.0 if self.cash_rate_annual is None else float(self.cash_rate_annual)
            )

        if self.borrow_rate_annual is not None:
            borrow_rate: RateInput = float(self.borrow_rate_annual)
        elif self.borrow_rate_spread is not None:
            borrow_rate = _add_rate_spread(
                cash_rate,
                spread=self.borrow_rate_spread,
            )
        else:
            borrow_rate = 0.0

        return MarginPolicy(
            maintenance_margin_ratio=self.maintenance_margin_ratio,
            margin_call_grace_days=self.margin_call_grace_days,
            liquidation_mode=self.liquidation_mode,  # type: ignore[arg-type]
            liquidation_buffer_ratio=self.liquidation_buffer_ratio,
            apply_financing=self.apply_financing,
            cash_rate_annual=cash_rate,
            borrow_rate_annual=borrow_rate,
            trading_days_per_year=self.trading_days_per_year,
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
    margin_policy_spec: MarginPolicySpec | None = None
    modeling: ModelingConfig = field(default_factory=ModelingConfig)

    def __post_init__(self) -> None:
        if (
            self.margin_policy_spec is not None
            and self.broker.margin.policy is not None
        ):
            raise ValueError(
                "broker.margin.policy and margin_policy_spec are mutually exclusive"
            )
        if (
            self.margin_policy_spec is not None
            and self.margin_policy_spec.cash_rate_source == "data_rates"
            and self.data.rates is None
        ):
            raise ValueError(
                "cash_rate_source='data_rates' requires data.rates in the workflow"
            )

    def to_backtest_run_config(
        self,
        *,
        data_rates: RateInput | None = None,
    ) -> BacktestRunConfig:
        """Project this workflow spec onto the existing engine run config."""
        broker = self.broker
        if self.margin_policy_spec is not None:
            broker = BrokerConfig(
                margin=MarginConfig(
                    model=self.broker.margin.model,
                    policy=self.margin_policy_spec.resolve(data_rates=data_rates),
                )
            )
        return BacktestRunConfig(
            account=self.account,
            execution=self.execution,
            broker=broker,
            modeling=self.modeling,
            start_date=self.run.start_date,
            end_date=self.run.end_date,
        )
