"""Backtest-oriented convenience loaders for canonical market-data inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from volatility_trading.backtesting.data_adapters import (
    OptionsChainAdapter,
    OratsOptionsChainAdapter,
    normalize_options_chain,
)
from volatility_trading.config.paths import PROC_FRED_RATES, PROC_YFINANCE_TIME_SERIES
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_fred_rates,
    read_options_chain,
    read_yfinance_time_series,
)


def canonicalize_options_chain_for_backtest(
    chain: object,
    *,
    adapter: OptionsChainAdapter,
) -> pd.DataFrame:
    """Normalize one raw options chain into canonical long pandas format."""
    return normalize_options_chain(chain, adapter=adapter)


def load_orats_options_chain_for_backtest(
    ticker: str,
    *,
    proc_root: Path | str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dte_min: float | None = None,
    dte_max: float | None = None,
) -> pd.DataFrame:
    """Load one processed ORATS options chain as canonical long pandas data."""
    if proc_root is None:
        wide = read_options_chain(ticker)
    else:
        wide = read_options_chain(ticker, proc_root=proc_root)
    long = options_chain_wide_to_long(wide).collect()
    options = canonicalize_options_chain_for_backtest(
        long,
        adapter=OratsOptionsChainAdapter(),
    )
    return filter_options_chain_for_backtest(
        options,
        start=start,
        end=end,
        dte_min=dte_min,
        dte_max=dte_max,
    )


def filter_options_chain_for_backtest(
    options_chain: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dte_min: float | None = None,
    dte_max: float | None = None,
    dte_col: str = "dte",
) -> pd.DataFrame:
    """Apply common date-window and DTE filters to a canonical options chain."""
    options = options_chain

    if start is not None:
        options = options.loc[options.index >= pd.Timestamp(start)]
    if end is not None:
        options = options.loc[options.index <= pd.Timestamp(end)]

    if dte_min is not None or dte_max is not None:
        if dte_col not in options.columns:
            raise ValueError(
                f"options_chain must contain '{dte_col}' for DTE filtering"
            )
        dte = options[dte_col]
        if dte_min is not None:
            options = options.loc[dte >= float(dte_min)]
            dte = options[dte_col]
        if dte_max is not None:
            options = options.loc[dte <= float(dte_max)]

    return options.sort_index()


def load_fred_rate_series(
    column: str,
    *,
    proc_root: Path | str = PROC_FRED_RATES,
    date_col: str = "date",
    as_decimal: bool = True,
) -> pd.Series:
    """Load one processed FRED rate column as an indexed pandas series."""
    resolved_root = _resolve_fred_rates_proc_root(proc_root)
    df = read_fred_rates(
        proc_root=resolved_root,
        columns=[date_col, column],
    ).to_pandas()
    df[date_col] = pd.to_datetime(df[date_col])
    series = df.set_index(date_col)[column].astype(float).sort_index()
    if as_decimal:
        series = series.div(100.0)
    series.name = column
    return series


def load_yfinance_close_series(
    ticker: str,
    *,
    proc_root: Path | str = PROC_YFINANCE_TIME_SERIES,
    date_col: str = "date",
    close_col: str = "close",
) -> pd.Series:
    """Load one processed yfinance close-price series as pandas."""
    resolved_root = _resolve_yfinance_time_series_proc_root(proc_root)
    df = read_yfinance_time_series(
        proc_root=resolved_root,
        columns=[date_col, close_col],
        tickers=[ticker],
    ).to_pandas()
    df[date_col] = pd.to_datetime(df[date_col])
    series = df.set_index(date_col)[close_col].astype(float).sort_index()
    series.name = close_col
    return series


def spot_series_from_options_chain(
    options_chain: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    spot_col: str = "spot_price",
) -> pd.Series:
    """Derive one spot-price series from a canonical options chain."""
    if spot_col not in options_chain.columns:
        raise ValueError(f"options_chain must contain '{spot_col}'")

    if date_col in options_chain.columns:
        series = (
            options_chain.groupby(date_col, sort=True)[spot_col].first().astype(float)
        )
    else:
        series = (
            options_chain.groupby(level=0, sort=True)[spot_col].first().astype(float)
        )
    series.index = pd.to_datetime(series.index)
    series.index.name = date_col
    series.name = spot_col
    return series.sort_index()


def _resolve_fred_rates_proc_root(proc_root: Path | str) -> Path:
    root = Path(proc_root)
    if (root / "fred_rates.parquet").exists():
        return root

    rates_root = root / "rates"
    if (rates_root / "fred_rates.parquet").exists():
        return rates_root

    return root


def _resolve_yfinance_time_series_proc_root(proc_root: Path | str) -> Path:
    root = Path(proc_root)
    if (root / "yfinance_time_series.parquet").exists():
        return root

    time_series_root = root / "time_series"
    if (time_series_root / "yfinance_time_series.parquet").exists():
        return time_series_root

    return root
