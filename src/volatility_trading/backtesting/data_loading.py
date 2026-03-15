"""Backtest-oriented convenience loaders for canonical market-data inputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import polars as pl

from volatility_trading.backtesting.data_adapters.options_chain_adapters import (
    CanonicalOptionsChainAdapter,
    OptionsChainAdapter,
    normalize_options_chain,
)
from volatility_trading.config.paths import (
    PROC_FRED_RATES,
    PROC_ORATS_DAILY_FEATURES,
    PROC_YFINANCE_TIME_SERIES,
)
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_daily_features,
    read_fred_rates,
    read_yfinance_time_series,
    scan_options_chain,
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
    long = _load_processed_orats_options_long_frame(
        ticker,
        proc_root=proc_root,
        start=start,
        end=end,
        dte_min=dte_min,
        dte_max=dte_max,
    )
    return canonicalize_options_chain_for_backtest(
        long,
        adapter=CanonicalOptionsChainAdapter(),
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


def _load_processed_orats_options_long_frame(
    ticker: str,
    *,
    proc_root: Path | str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dte_min: float | None = None,
    dte_max: float | None = None,
) -> pd.DataFrame:
    """Load one processed ORATS chain into long pandas format with pushdown filters."""
    wide = _scan_processed_orats_options_chain(
        ticker,
        proc_root=proc_root,
        start=start,
        end=end,
        dte_min=dte_min,
        dte_max=dte_max,
    )
    return options_chain_wide_to_long(wide).collect().to_pandas()


def _scan_processed_orats_options_chain(
    ticker: str,
    *,
    proc_root: Path | str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dte_min: float | None = None,
    dte_max: float | None = None,
) -> pl.LazyFrame:
    """Scan one processed ORATS chain and push supported source filters early."""
    if proc_root is None:
        wide = scan_options_chain(ticker)
    else:
        wide = scan_options_chain(ticker, proc_root=proc_root)

    if start is not None:
        wide = wide.filter(pl.col("trade_date") >= pl.lit(pd.Timestamp(start)))
    if end is not None:
        wide = wide.filter(pl.col("trade_date") <= pl.lit(pd.Timestamp(end)))
    if dte_min is not None:
        wide = wide.filter(pl.col("dte") >= float(dte_min))
    if dte_max is not None:
        wide = wide.filter(pl.col("dte") <= float(dte_max))

    return wide


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


def load_daily_features_frame(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_DAILY_FEATURES,
    columns: Sequence[str] | None = None,
    date_col: str = "trade_date",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load one processed daily-features panel as an indexed pandas frame."""
    requested_columns = None
    if columns is not None:
        requested_columns = [date_col, *columns]
        requested_columns = list(dict.fromkeys(requested_columns))

    frame = read_daily_features(
        ticker,
        proc_root=proc_root,
        columns=requested_columns,
    ).to_pandas()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        range_text = (
            f" in range {start}:{end}" if start is not None or end is not None else ""
        )
        raise ValueError(f"No daily-features rows for {ticker}{range_text}")
    return frame


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
