"""Internal source-loader registries for workflow-runner data providers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd

from volatility_trading.backtesting.data_adapters import CanonicalOptionsChainAdapter
from volatility_trading.backtesting.data_contracts import OptionsMarketData
from volatility_trading.backtesting.data_loading import (
    canonicalize_options_chain_for_backtest,
    filter_options_chain_for_backtest,
)
from volatility_trading.backtesting.rates import RateInput
from volatility_trading.datasets import (
    fred_rates_path,
    options_chain_wide_to_long,
    read_daily_features,
    read_fred_rates,
    read_options_chain,
    read_yfinance_time_series,
)

if TYPE_CHECKING:
    from .workflow_types import (
        BacktestWorkflowSpec,
        FeaturesSourceSpec,
        OptionsSourceSpec,
        RatesSourceSpec,
        SeriesSourceSpec,
    )

OptionsSourceLoader: TypeAlias = Callable[[Any], OptionsMarketData]
FeaturesSourceLoader: TypeAlias = Callable[[Any], pd.DataFrame]
SeriesSourceLoader: TypeAlias = Callable[[Any], pd.Series]
RatesSourceLoader: TypeAlias = Callable[[Any, Any], RateInput]


def _load_orats_options_market(spec: OptionsSourceSpec) -> OptionsMarketData:
    """Load one ORATS options chain into canonical long pandas market data."""
    if spec.proc_root is None:
        wide = read_options_chain(spec.ticker)
    else:
        wide = read_options_chain(spec.ticker, proc_root=spec.proc_root)
    long = options_chain_wide_to_long(wide).collect().to_pandas()
    options = canonicalize_options_chain_for_backtest(
        long,
        adapter=CanonicalOptionsChainAdapter(),
    )
    options = filter_options_chain_for_backtest(
        options,
        dte_min=spec.dte_min,
        dte_max=spec.dte_max,
    )

    return OptionsMarketData(
        chain=options,
        symbol=spec.symbol or spec.ticker,
        default_contract_multiplier=spec.default_contract_multiplier,
    )


def _load_orats_features_frame(spec: FeaturesSourceSpec) -> pd.DataFrame:
    """Load one ORATS daily-features source into pandas."""
    if spec.proc_root is None:
        frame = read_daily_features(spec.ticker).to_pandas()
    else:
        frame = read_daily_features(spec.ticker, proc_root=spec.proc_root).to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").sort_index()


def _load_yfinance_series(spec: SeriesSourceSpec) -> pd.Series:
    """Load one market time series from processed yfinance data."""
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


def _load_constant_rate_input(
    spec: RatesSourceSpec,
    _workflow: BacktestWorkflowSpec,
) -> RateInput:
    """Resolve one constant-rate input."""
    rate = spec.constant_rate
    if rate is None:
        raise ValueError("constant-rate source requires constant_rate to be finite")
    return float(rate)


def _load_fred_rate_input(
    spec: RatesSourceSpec,
    workflow: BacktestWorkflowSpec,
) -> RateInput:
    """Resolve one FRED rates source into a sliced rate series."""
    fred_proc_root = _resolve_fred_rates_proc_root(spec.proc_root)
    if spec.proc_root is None:
        frame = read_fred_rates().to_pandas()
    else:
        frame = read_fred_rates(proc_root=fred_proc_root).to_pandas()
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
    # Processed FRED rates are stored in percentage units (e.g. 2.32 for 2.32%).
    series = series / 100.0
    return slice_series_to_run_window(series, workflow=workflow)


def _resolve_fred_rates_proc_root(proc_root: Path | None) -> Path | None:
    """Accept either a FRED source root or a FRED rates-domain root."""
    if proc_root is None:
        return None
    root = Path(proc_root)
    if fred_rates_path(root).exists():
        return root
    domain_root = root / "rates"
    if fred_rates_path(domain_root).exists():
        return domain_root
    return root


OPTIONS_SOURCE_LOADERS: dict[str, OptionsSourceLoader] = {
    "orats": _load_orats_options_market,
}
FEATURES_SOURCE_LOADERS: dict[str, FeaturesSourceLoader] = {
    "orats": _load_orats_features_frame,
}
SERIES_SOURCE_LOADERS: dict[str, SeriesSourceLoader] = {
    "yfinance": _load_yfinance_series,
}
RATES_SOURCE_LOADERS: dict[str, RatesSourceLoader] = {
    "constant": _load_constant_rate_input,
    "fred": _load_fred_rate_input,
}

OPTIONS_SOURCE_PROVIDERS = tuple(sorted(OPTIONS_SOURCE_LOADERS))
FEATURES_SOURCE_PROVIDERS = tuple(sorted(FEATURES_SOURCE_LOADERS))
SERIES_SOURCE_PROVIDERS = tuple(sorted(SERIES_SOURCE_LOADERS))
RATES_SOURCE_PROVIDERS = tuple(sorted(RATES_SOURCE_LOADERS))


def load_options_market(spec: OptionsSourceSpec) -> OptionsMarketData:
    """Dispatch one typed options source spec through the provider registry."""
    return OPTIONS_SOURCE_LOADERS[spec.provider](spec)


def load_features_frame(spec: FeaturesSourceSpec | None) -> pd.DataFrame | None:
    """Dispatch one optional features source spec through the provider registry."""
    if spec is None:
        return None
    return FEATURES_SOURCE_LOADERS[spec.provider](spec)


def load_series(spec: SeriesSourceSpec | None) -> pd.Series | None:
    """Dispatch one optional market series source through the provider registry."""
    if spec is None:
        return None
    return SERIES_SOURCE_LOADERS[spec.provider](spec)


def load_rate_input(
    spec: RatesSourceSpec | None,
    *,
    workflow: BacktestWorkflowSpec,
) -> RateInput:
    """Dispatch one optional rates source through the provider registry."""
    if spec is None:
        return 0.0
    return RATES_SOURCE_LOADERS[spec.provider](spec, workflow)


def slice_series_to_run_window(
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
