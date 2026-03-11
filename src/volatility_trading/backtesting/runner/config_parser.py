"""Parse nested workflow config mappings into typed runner specs."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

from volatility_trading.backtesting.config import (
    AccountConfig,
    BrokerConfig,
    ExecutionConfig,
    ModelingConfig,
)
from volatility_trading.backtesting.options_engine.lifecycle import (
    BidAskFeeOptionExecutionModel,
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionModel,
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

OptionExecutionFactory: TypeAlias = Callable[..., OptionExecutionModel]
HedgeExecutionFactory: TypeAlias = Callable[..., HedgeExecutionModel]

_OPTION_EXECUTION_FACTORIES: dict[str, OptionExecutionFactory] = {
    "bid_ask_fee": BidAskFeeOptionExecutionModel,
    "mid_no_cost": MidNoCostOptionExecutionModel,
}
_HEDGE_EXECUTION_FACTORIES: dict[str, HedgeExecutionFactory] = {
    "fixed_bps": FixedBpsHedgeExecutionModel,
    "mid_no_cost": MidNoCostHedgeExecutionModel,
}


def _expect_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    """Require a mapping payload at one parser boundary."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _ensure_keys(
    payload: Mapping[str, Any],
    *,
    field_name: str,
    allowed: tuple[str, ...],
    required: tuple[str, ...] = (),
) -> None:
    """Validate allowed/required keys for one mapping payload."""
    missing = tuple(key for key in required if key not in payload)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{field_name} is missing required keys: {missing_text}")

    unexpected = tuple(sorted(set(payload) - set(allowed)))
    if unexpected:
        unexpected_text = ", ".join(unexpected)
        allowed_text = ", ".join(allowed)
        raise ValueError(
            f"{field_name} contains unsupported keys: {unexpected_text}. "
            f"Allowed keys: {allowed_text}."
        )


def _resolve_path(value: str | Path | None) -> Path | None:
    """Resolve optional path-like config strings."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded)


def _accepted_params_text(factory: Callable[..., Any]) -> str:
    """Describe accepted constructor kwargs for clear config errors."""
    accepted = tuple(inspect.signature(factory).parameters)
    return ", ".join(accepted) if accepted else "none"


def _build_model_instance(
    *,
    section_name: str,
    model_name: str,
    params: Mapping[str, Any],
    factories: Mapping[str, Callable[..., Any]],
) -> Any:
    """Instantiate one named model from a registry and kwargs mapping."""
    factory = factories.get(model_name)
    if factory is None:
        available = ", ".join(sorted(factories))
        raise ValueError(
            f"Unknown {section_name} model '{model_name}'. "
            f"Available models: {available}."
        )

    try:
        inspect.signature(factory).bind(**params)
    except TypeError as exc:
        accepted = _accepted_params_text(factory)
        raise ValueError(
            f"Invalid parameters for {section_name} model '{model_name}': {exc}. "
            f"Accepted parameters: {accepted}."
        ) from exc
    return factory(**dict(params))


def parse_workflow_config(config: Mapping[str, Any]) -> BacktestWorkflowSpec:
    """Parse one merged config mapping into a typed workflow spec.

    This parser intentionally handles only the first runner slice:
    - data sources
    - strategy/signal registry specs
    - account config
    - execution model config
    - run window
    - reporting config

    `broker` and `modeling` currently accept only empty mappings and otherwise
    stay on existing defaults.
    """
    payload = _expect_mapping(config, field_name="workflow config")
    _ensure_keys(
        payload,
        field_name="workflow config",
        allowed=(
            "account",
            "broker",
            "data",
            "execution",
            "modeling",
            "reporting",
            "run",
            "strategy",
        ),
        required=("data", "strategy"),
    )

    return BacktestWorkflowSpec(
        data=_parse_data_sources_spec(payload["data"]),
        strategy=_parse_named_strategy_preset_spec(payload["strategy"]),
        run=_parse_run_window_spec(payload.get("run")),
        reporting=_parse_reporting_spec(payload.get("reporting")),
        account=_parse_account_config(payload.get("account")),
        execution=_parse_execution_config(payload.get("execution")),
        broker=_parse_broker_config(payload.get("broker")),
        modeling=_parse_modeling_config(payload.get("modeling")),
    )


def _parse_data_sources_spec(payload: Any) -> BacktestDataSourcesSpec:
    """Parse one nested data-sources config mapping."""
    mapping = _expect_mapping(payload, field_name="data")
    _ensure_keys(
        mapping,
        field_name="data",
        allowed=("benchmark", "features", "hedge", "options", "rates"),
        required=("options",),
    )
    return BacktestDataSourcesSpec(
        options=_parse_options_source_spec(mapping["options"]),
        features=_parse_features_source_spec(mapping.get("features")),
        hedge=_parse_series_source_spec(mapping.get("hedge"), field_name="data.hedge"),
        benchmark=_parse_series_source_spec(
            mapping.get("benchmark"),
            field_name="data.benchmark",
        ),
        rates=_parse_rates_source_spec(mapping.get("rates")),
    )


def _parse_options_source_spec(payload: Any) -> OptionsSourceSpec:
    """Parse one options source mapping."""
    mapping = _expect_mapping(payload, field_name="data.options")
    _ensure_keys(
        mapping,
        field_name="data.options",
        allowed=(
            "adapter_name",
            "default_contract_multiplier",
            "proc_root",
            "provider",
            "symbol",
            "ticker",
        ),
        required=("ticker",),
    )
    return OptionsSourceSpec(
        ticker=str(mapping["ticker"]),
        provider=str(mapping.get("provider", "orats")),
        proc_root=_resolve_path(mapping.get("proc_root")),
        adapter_name=(
            None
            if mapping.get("adapter_name") is None
            else str(mapping["adapter_name"])
        ),
        symbol=None if mapping.get("symbol") is None else str(mapping["symbol"]),
        default_contract_multiplier=float(
            mapping.get("default_contract_multiplier", 100.0)
        ),
    )


def _parse_features_source_spec(payload: Any) -> FeaturesSourceSpec | None:
    """Parse one optional features source mapping."""
    if payload is None:
        return None
    mapping = _expect_mapping(payload, field_name="data.features")
    _ensure_keys(
        mapping,
        field_name="data.features",
        allowed=("proc_root", "provider", "ticker"),
        required=("ticker",),
    )
    return FeaturesSourceSpec(
        ticker=str(mapping["ticker"]),
        provider=str(mapping.get("provider", "orats")),
        proc_root=_resolve_path(mapping.get("proc_root")),
    )


def _parse_series_source_spec(
    payload: Any,
    *,
    field_name: str,
) -> SeriesSourceSpec | None:
    """Parse one optional hedge/benchmark source mapping."""
    if payload is None:
        return None
    mapping = _expect_mapping(payload, field_name=field_name)
    _ensure_keys(
        mapping,
        field_name=field_name,
        allowed=(
            "contract_multiplier",
            "price_column",
            "proc_root",
            "provider",
            "symbol",
            "ticker",
        ),
        required=("ticker",),
    )
    return SeriesSourceSpec(
        ticker=str(mapping["ticker"]),
        provider=str(mapping.get("provider", "yfinance")),
        proc_root=_resolve_path(mapping.get("proc_root")),
        price_column=str(mapping.get("price_column", "close")),
        symbol=None if mapping.get("symbol") is None else str(mapping["symbol"]),
        contract_multiplier=float(mapping.get("contract_multiplier", 1.0)),
    )


def _parse_rates_source_spec(payload: Any) -> RatesSourceSpec | None:
    """Parse one optional rates source mapping."""
    if payload is None:
        return None
    mapping = _expect_mapping(payload, field_name="data.rates")
    _ensure_keys(
        mapping,
        field_name="data.rates",
        allowed=("column", "constant_rate", "proc_root", "provider", "series_id"),
    )
    return RatesSourceSpec(
        provider=str(mapping.get("provider", "constant")),
        constant_rate=(
            None
            if mapping.get("constant_rate") is None
            else float(mapping["constant_rate"])
        ),
        series_id=(
            None if mapping.get("series_id") is None else str(mapping["series_id"])
        ),
        column=None if mapping.get("column") is None else str(mapping["column"]),
        proc_root=_resolve_path(mapping.get("proc_root")),
    )


def _parse_named_signal_spec(payload: Any) -> NamedSignalSpec:
    """Parse one named signal mapping."""
    mapping = _expect_mapping(payload, field_name="strategy.signal")
    _ensure_keys(
        mapping,
        field_name="strategy.signal",
        allowed=("name", "params"),
        required=("name",),
    )
    params = mapping.get("params", {})
    return NamedSignalSpec(
        name=str(mapping["name"]),
        params=_expect_mapping(params, field_name="strategy.signal.params"),
    )


def _parse_named_strategy_preset_spec(payload: Any) -> NamedStrategyPresetSpec:
    """Parse one named strategy preset mapping."""
    mapping = _expect_mapping(payload, field_name="strategy")
    _ensure_keys(
        mapping,
        field_name="strategy",
        allowed=("name", "params", "signal"),
        required=("name",),
    )
    params = mapping.get("params", {})
    signal_payload = mapping.get("signal")
    return NamedStrategyPresetSpec(
        name=str(mapping["name"]),
        signal=(
            None if signal_payload is None else _parse_named_signal_spec(signal_payload)
        ),
        params=_expect_mapping(params, field_name="strategy.params"),
    )


def _parse_run_window_spec(payload: Any) -> RunWindowSpec:
    """Parse one optional run-window mapping."""
    if payload is None:
        return RunWindowSpec()
    mapping = _expect_mapping(payload, field_name="run")
    _ensure_keys(
        mapping,
        field_name="run",
        allowed=("end_date", "start_date"),
    )
    return RunWindowSpec(
        start_date=mapping.get("start_date"),
        end_date=mapping.get("end_date"),
    )


def _parse_reporting_spec(payload: Any) -> ReportingSpec:
    """Parse one optional reporting config mapping."""
    if payload is None:
        return ReportingSpec()
    mapping = _expect_mapping(payload, field_name="reporting")
    _ensure_keys(
        mapping,
        field_name="reporting",
        allowed=(
            "benchmark_name",
            "include_component_plots",
            "include_dashboard_plot",
            "output_root",
            "run_id",
            "save_report_bundle",
        ),
    )
    return ReportingSpec(
        output_root=_resolve_path(mapping.get("output_root"))
        or Path("reports/backtests"),
        run_id=None if mapping.get("run_id") is None else str(mapping["run_id"]),
        benchmark_name=(
            None
            if mapping.get("benchmark_name") is None
            else str(mapping["benchmark_name"])
        ),
        include_dashboard_plot=bool(mapping.get("include_dashboard_plot", True)),
        include_component_plots=bool(mapping.get("include_component_plots", False)),
        save_report_bundle=bool(mapping.get("save_report_bundle", True)),
    )


def _parse_account_config(payload: Any) -> AccountConfig:
    """Parse one optional account config mapping."""
    if payload is None:
        return AccountConfig()
    mapping = _expect_mapping(payload, field_name="account")
    _ensure_keys(
        mapping,
        field_name="account",
        allowed=("initial_capital",),
    )
    return AccountConfig(
        initial_capital=float(mapping.get("initial_capital", 100_000.0))
    )


def _parse_execution_config(payload: Any) -> ExecutionConfig:
    """Parse one optional execution config mapping."""
    if payload is None:
        return ExecutionConfig()
    mapping = _expect_mapping(payload, field_name="execution")
    _ensure_keys(
        mapping,
        field_name="execution",
        allowed=("hedge", "option"),
    )

    option_model = _parse_execution_model_section(
        mapping.get("option"),
        field_name="execution.option",
        default_name="bid_ask_fee",
        factories=_OPTION_EXECUTION_FACTORIES,
    )
    hedge_model = _parse_execution_model_section(
        mapping.get("hedge"),
        field_name="execution.hedge",
        default_name="fixed_bps",
        factories=_HEDGE_EXECUTION_FACTORIES,
    )
    return ExecutionConfig(
        option_execution_model=option_model,
        hedge_execution_model=hedge_model,
    )


def _parse_execution_model_section(
    payload: Any,
    *,
    field_name: str,
    default_name: str,
    factories: Mapping[str, Callable[..., Any]],
) -> Any:
    """Parse one execution-model subsection into a concrete model instance."""
    if payload is None:
        return _build_model_instance(
            section_name=field_name,
            model_name=default_name,
            params={},
            factories=factories,
        )
    mapping = _expect_mapping(payload, field_name=field_name)
    _ensure_keys(
        mapping,
        field_name=field_name,
        allowed=("model", "params"),
    )
    params = mapping.get("params", {})
    return _build_model_instance(
        section_name=field_name,
        model_name=str(mapping.get("model", default_name)),
        params=_expect_mapping(params, field_name=f"{field_name}.params"),
        factories=factories,
    )


def _parse_broker_config(payload: Any) -> BrokerConfig:
    """Parse broker config for the current workflow-parser slice.

    Margin-model and policy parsing is deferred to a later commit. For now, the
    section can be omitted or present as an empty mapping.
    """
    if payload is None:
        return BrokerConfig()
    mapping = _expect_mapping(payload, field_name="broker")
    _ensure_keys(mapping, field_name="broker", allowed=())
    return BrokerConfig()


def _parse_modeling_config(payload: Any) -> ModelingConfig:
    """Parse modeling config for the current workflow-parser slice.

    Runtime pricing/risk model selection is deferred to a later commit. For
    now, the section can be omitted or present as an empty mapping.
    """
    if payload is None:
        return ModelingConfig()
    mapping = _expect_mapping(payload, field_name="modeling")
    _ensure_keys(mapping, field_name="modeling", allowed=())
    return ModelingConfig()
