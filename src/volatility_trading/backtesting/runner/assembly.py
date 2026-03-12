"""Resolve typed workflow specs into concrete backtest runtime inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting.config import BacktestRunConfig
from volatility_trading.backtesting.data_adapters import (
    CanonicalOptionsChainAdapter,
    OptionsChainAdapter,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    YfinanceOptionsChainAdapter,
)
from volatility_trading.backtesting.data_contracts import (
    HedgeMarketData,
    OptionsBacktestDataBundle,
    OptionsMarketData,
)
from volatility_trading.backtesting.options_engine.specs import StrategySpec
from volatility_trading.backtesting.rates import RateInput
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_daily_features,
    read_fred_rates,
    read_options_chain,
    read_yfinance_time_series,
)

from .registry import build_strategy_preset
from .workflow_types import (
    BacktestWorkflowSpec,
    FeaturesSourceSpec,
    OptionsSourceSpec,
    RatesSourceSpec,
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
    run_config = workflow.to_backtest_run_config()
    _validate_workflow_compatibility(
        workflow=workflow,
        strategy=strategy,
        run_config=run_config,
    )
    options_market = _load_options_market(workflow.data.options)
    features = _load_features_frame(workflow.data.features)
    hedge_market = _load_hedge_market(workflow.data.hedge)
    benchmark = _slice_series_to_run_window(
        _load_series(workflow.data.benchmark),
        workflow=workflow,
    )
    risk_free_rate = _resolve_risk_free_rate(
        workflow.data.rates,
        workflow=workflow,
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


def _load_options_market(spec: OptionsSourceSpec) -> OptionsMarketData:
    """Load and normalize one options market source into canonical long pandas."""
    if spec.provider != "orats":
        raise ValueError(f"Unsupported options source provider: {spec.provider}")
    adapter = _resolve_options_adapter(spec)

    if spec.proc_root is None:
        wide = read_options_chain(spec.ticker)
    else:
        wide = read_options_chain(spec.ticker, proc_root=spec.proc_root)
    long = options_chain_wide_to_long(wide).collect().to_pandas()
    long["trade_date"] = pd.to_datetime(long["trade_date"])
    long = long.set_index("trade_date").sort_index()

    return OptionsMarketData(
        chain=long,
        symbol=spec.symbol or spec.ticker,
        default_contract_multiplier=spec.default_contract_multiplier,
        options_adapter=adapter,
    )


def _resolve_options_adapter(spec: OptionsSourceSpec) -> OptionsChainAdapter:
    """Resolve one built-in adapter name for canonicalized options data."""
    if spec.adapter_name is None:
        return CanonicalOptionsChainAdapter()

    adapter_name = spec.adapter_name.lower()
    factories: dict[str, Callable[[], OptionsChainAdapter]] = {
        "canonical": CanonicalOptionsChainAdapter,
        "orats": OratsOptionsChainAdapter,
        "optionsdx": OptionsDxOptionsChainAdapter,
        "yfinance": YfinanceOptionsChainAdapter,
    }
    factory = factories.get(adapter_name)
    if factory is None:
        available = ", ".join(sorted(factories))
        raise ValueError(
            f"Unknown options adapter_name '{spec.adapter_name}'. "
            f"Available built-in adapters: {available}."
        )
    return factory()


def _load_features_frame(spec: FeaturesSourceSpec | None) -> pd.DataFrame | None:
    """Load one optional daily-features source into pandas."""
    if spec is None:
        return None
    if spec.provider != "orats":
        raise ValueError(f"Unsupported features source provider: {spec.provider}")

    if spec.proc_root is None:
        frame = read_daily_features(spec.ticker).to_pandas()
    else:
        frame = read_daily_features(spec.ticker, proc_root=spec.proc_root).to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").sort_index()


def _load_hedge_market(spec: SeriesSourceSpec | None) -> HedgeMarketData | None:
    """Load one optional hedge market source into backtesting market data."""
    series = _load_series(spec)
    if spec is None or series is None:
        return None
    return HedgeMarketData(
        mid=series,
        symbol=spec.symbol or spec.ticker,
        contract_multiplier=spec.contract_multiplier,
    )


def _load_series(spec: SeriesSourceSpec | None) -> pd.Series | None:
    """Load one optional market time series from processed yfinance data."""
    if spec is None:
        return None
    if spec.provider != "yfinance":
        raise ValueError(f"Unsupported series source provider: {spec.provider}")

    if spec.proc_root is None:
        frame = read_yfinance_time_series(tickers=[spec.ticker]).to_pandas()
    else:
        frame = read_yfinance_time_series(
            proc_root=spec.proc_root,
            tickers=[spec.ticker],
        ).to_pandas()
    if frame.empty:
        raise ValueError(f"No yfinance rows found for ticker {spec.ticker}")
    if spec.price_column not in frame.columns:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Series source {spec.ticker} requires price_column "
            f"'{spec.price_column}', but available columns are: {available}"
        )

    frame["date"] = pd.to_datetime(frame["date"])
    series = pd.Series(
        pd.to_numeric(frame[spec.price_column], errors="coerce").values,
        index=pd.DatetimeIndex(frame["date"]),
        name=spec.symbol or spec.ticker,
    )
    return series.groupby(level=0).last().sort_index()


def _resolve_risk_free_rate(
    spec: RatesSourceSpec | None,
    *,
    workflow: BacktestWorkflowSpec,
) -> RateInput:
    """Resolve one optional rates source into the reporting/margin rate input."""
    if spec is None:
        return 0.0
    if spec.provider == "constant":
        return float(spec.constant_rate or 0.0)
    if spec.provider != "fred":
        raise ValueError(f"Unsupported rates source provider: {spec.provider}")

    if spec.proc_root is None:
        frame = read_fred_rates().to_pandas()
    else:
        frame = read_fred_rates(proc_root=spec.proc_root).to_pandas()
    column = spec.column or str(spec.series_id).lower()
    if column not in frame.columns:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"FRED rates source requires column '{column}', "
            f"but available columns are: {available}"
        )
    frame["date"] = pd.to_datetime(frame["date"])
    series = (
        pd.Series(
            pd.to_numeric(frame[column], errors="coerce").values,
            index=pd.DatetimeIndex(frame["date"]),
            name=column,
        )
        .groupby(level=0)
        .last()
        .sort_index()
    )
    return _slice_series_to_run_window(series, workflow=workflow)


def _slice_series_to_run_window(
    series: pd.Series | None,
    *,
    workflow: BacktestWorkflowSpec,
) -> pd.Series | None:
    """Slice an optional series to the workflow run window."""
    if series is None:
        return None
    out = series.sort_index()
    if workflow.run.start_date is not None:
        out = out.loc[out.index >= workflow.run.start_date]
    if workflow.run.end_date is not None:
        out = out.loc[out.index <= workflow.run.end_date]
    return out


def _resolve_benchmark_name(workflow: BacktestWorkflowSpec) -> str | None:
    """Resolve the effective benchmark label for reporting."""
    if workflow.data.benchmark is None:
        return None
    if workflow.reporting.benchmark_name is not None:
        return workflow.reporting.benchmark_name
    return workflow.data.benchmark.symbol or workflow.data.benchmark.ticker
