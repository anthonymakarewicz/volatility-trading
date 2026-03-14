"""Resolve typed workflow specs into concrete backtest runtime inputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.config import BacktestRunConfig
from volatility_trading.backtesting.data_contracts import (
    HedgeMarketData,
    OptionsBacktestDataBundle,
)
from volatility_trading.backtesting.options_engine.specs import StrategySpec
from volatility_trading.backtesting.rates import RateInput

from .registry import build_strategy_preset
from .source_loaders import (
    load_features_frame,
    load_options_market,
    load_rate_input,
    load_series,
    slice_series_to_run_window,
)
from .workflow_types import (
    BacktestWorkflowSpec,
    SeriesSourceSpec,
)


@dataclass(frozen=True)
class ResolvedWorkflowInputs:
    """Concrete in-memory inputs assembled from one typed workflow spec."""

    workflow: BacktestWorkflowSpec
    strategy: StrategySpec
    data: OptionsBacktestDataBundle
    run_config: BacktestRunConfig
    benchmark: pd.Series | None
    benchmark_name: str | None
    risk_free_rate: RateInput


def assemble_workflow_inputs(
    workflow: BacktestWorkflowSpec,
) -> ResolvedWorkflowInputs:
    """Resolve sources and configs into concrete runtime inputs."""
    strategy = build_strategy_preset(workflow.strategy)
    options_market = load_options_market(workflow.data.options)
    features = load_features_frame(workflow.data.features)
    hedge_market = _load_hedge_market(workflow.data.hedge)
    benchmark = slice_series_to_run_window(
        load_series(workflow.data.benchmark),
        workflow=workflow,
    )
    risk_free_rate = load_rate_input(
        workflow.data.rates,
        workflow=workflow,
    )
    run_config = workflow.to_backtest_run_config(
        data_rates=None if workflow.data.rates is None else risk_free_rate
    )
    _validate_workflow_compatibility(
        workflow=workflow,
        strategy=strategy,
        run_config=run_config,
    )
    data_bundle = OptionsBacktestDataBundle(
        options_market=options_market,
        features=features,
        hedge_market=hedge_market,
    )
    return ResolvedWorkflowInputs(
        workflow=workflow,
        strategy=strategy,
        data=data_bundle,
        run_config=run_config,
        benchmark=benchmark,
        benchmark_name=_resolve_benchmark_name(workflow),
        risk_free_rate=risk_free_rate,
    )


def _validate_workflow_compatibility(
    *,
    workflow: BacktestWorkflowSpec,
    strategy: StrategySpec,
    run_config: BacktestRunConfig,
) -> None:
    """Reject workflow combinations that the engine requires upfront."""
    if (
        strategy.sizing.margin_budget_pct is not None
        and run_config.broker.margin.model is None
    ):
        raise ValueError(
            "strategy margin_budget_pct requires broker.margin.model in the workflow config"
        )
    if strategy.lifecycle.delta_hedge.enabled and workflow.data.hedge is None:
        raise ValueError(
            "enabled delta hedging requires data.hedge in the workflow config"
        )


def _load_hedge_market(spec: SeriesSourceSpec | None) -> HedgeMarketData | None:
    """Load one optional hedge market source into backtesting market data."""
    series = load_series(spec)
    if spec is None or series is None:
        return None
    return HedgeMarketData(
        mid=series,
        symbol=spec.symbol or spec.ticker,
        contract_multiplier=spec.contract_multiplier,
    )


def _resolve_benchmark_name(workflow: BacktestWorkflowSpec) -> str | None:
    """Resolve the effective benchmark label for reporting."""
    if workflow.data.benchmark is None:
        return None
    if workflow.reporting.benchmark_name is not None:
        return workflow.reporting.benchmark_name
    return workflow.data.benchmark.symbol or workflow.data.benchmark.ticker
