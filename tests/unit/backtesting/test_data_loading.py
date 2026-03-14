from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from volatility_trading.backtesting import (
    canonicalize_options_chain_for_backtest,
    data_loading,
    filter_options_chain_for_backtest,
    load_daily_features_frame,
    load_fred_rate_series,
    load_orats_options_chain_for_backtest,
    load_yfinance_close_series,
    spot_series_from_options_chain,
)
from volatility_trading.backtesting.data_adapters import ColumnMapOptionsChainAdapter


def test_canonicalize_options_chain_for_backtest_normalizes_with_adapter() -> None:
    raw = pd.DataFrame(
        {
            "quote_date": ["2024-01-03", "2024-01-02"],
            "expiration": ["2024-02-16", "2024-02-16"],
            "kind": ["call", "put"],
            "dte_value": [44.0, 45.0],
            "strike_value": [100.0, 100.0],
            "delta_value": [0.5, -0.5],
            "bid_value": [2.0, 1.8],
            "ask_value": [2.2, 2.0],
            "spot_value": [101.0, 100.0],
            "iv_value": [0.20, 0.21],
        }
    )
    adapter = ColumnMapOptionsChainAdapter(
        name="mapped",
        source_to_canonical={
            "quote_date": "trade_date",
            "expiration": "expiry_date",
            "kind": "option_type",
            "dte_value": "dte",
            "strike_value": "strike",
            "delta_value": "delta",
            "bid_value": "bid_price",
            "ask_value": "ask_price",
            "spot_value": "spot_price",
            "iv_value": "market_iv",
        },
    )

    options = canonicalize_options_chain_for_backtest(raw, adapter=adapter)

    assert isinstance(options.index, pd.DatetimeIndex)
    assert options.index.name == "trade_date"
    assert list(options["option_type"]) == ["P", "C"]
    assert list(options.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_load_orats_options_chain_for_backtest_returns_canonical_long_panel(
    monkeypatch,
) -> None:
    wide = pl.DataFrame(
        {
            "trade_date": ["2024-01-03"],
            "expiry_date": ["2024-02-16"],
            "dte": [44.0],
            "strike": [100.0],
            "spot_price": [101.0],
            "call_bid_price": [2.0],
            "call_ask_price": [2.2],
            "call_delta": [0.5],
            "put_bid_price": [1.8],
            "put_ask_price": [2.0],
            "put_delta": [-0.5],
        }
    )

    def fake_read_options_chain(ticker: str, *, proc_root=None, columns=None):
        assert ticker == "SPY"
        return wide

    monkeypatch.setattr(data_loading, "read_options_chain", fake_read_options_chain)

    options = load_orats_options_chain_for_backtest("SPY")

    assert isinstance(options.index, pd.DatetimeIndex)
    assert options.index.name == "trade_date"
    assert list(options["option_type"]) == ["C", "P"]
    assert {"expiry_date", "dte", "strike", "delta", "bid_price", "ask_price"} <= set(
        options.columns
    )


def test_load_orats_options_chain_for_backtest_applies_date_and_dte_filters(
    monkeypatch,
) -> None:
    wide = pl.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "expiry_date": ["2024-02-16", "2024-02-16", "2024-02-16"],
            "dte": [45.0, 44.0, 30.0],
            "strike": [100.0, 100.0, 100.0],
            "spot_price": [100.0, 101.0, 102.0],
            "call_bid_price": [2.0, 2.1, 2.2],
            "call_ask_price": [2.2, 2.3, 2.4],
            "call_delta": [0.5, 0.5, 0.5],
            "put_bid_price": [1.8, 1.9, 2.0],
            "put_ask_price": [2.0, 2.1, 2.2],
            "put_delta": [-0.5, -0.5, -0.5],
        }
    )

    monkeypatch.setattr(
        data_loading,
        "read_options_chain",
        lambda ticker, *, proc_root=None, columns=None: wide,
    )

    options = load_orats_options_chain_for_backtest(
        "SPY",
        start="2024-01-03",
        end="2024-01-04",
        dte_min=35.0,
        dte_max=44.0,
    )

    assert list(options.index.unique()) == list(pd.to_datetime(["2024-01-03"]))
    assert set(options["dte"]) == {44.0}


def test_filter_options_chain_for_backtest_rejects_dte_filter_without_dte_column() -> (
    None
):
    options = pd.DataFrame(
        {"strike": [100.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    options.index.name = "trade_date"

    with pytest.raises(ValueError, match="must contain 'dte'"):
        filter_options_chain_for_backtest(options, dte_min=10.0)


def test_load_fred_rate_series_accepts_source_root_and_scales_to_decimal(
    monkeypatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "fred"
    rates_root = source_root / "rates"
    rates_root.mkdir(parents=True)
    (rates_root / "fred_rates.parquet").touch()
    seen: dict[str, Path] = {}

    def fake_read_fred_rates(*, proc_root, columns=None):
        seen["proc_root"] = Path(proc_root)
        return pl.DataFrame(
            {
                "date": ["2024-01-03", "2024-01-02"],
                "dgs3mo": [1.5, 2.0],
            }
        )

    monkeypatch.setattr(data_loading, "read_fred_rates", fake_read_fred_rates)

    rf = load_fred_rate_series("dgs3mo", proc_root=source_root)

    assert seen["proc_root"] == rates_root
    assert list(rf.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert rf.iloc[0] == 0.02
    assert rf.iloc[1] == 0.015


def test_load_daily_features_frame_accepts_optional_columns_and_window(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_read_daily_features(ticker: str, *, proc_root, columns=None):
        seen["ticker"] = ticker
        seen["proc_root"] = Path(proc_root)
        seen["columns"] = tuple(columns) if columns is not None else None
        return pl.DataFrame(
            {
                "trade_date": ["2024-01-03", "2024-01-02"],
                "iv_dlt25_30d": [0.31, 0.30],
                "iv_dlt75_30d": [0.20, 0.19],
            }
        )

    monkeypatch.setattr(data_loading, "read_daily_features", fake_read_daily_features)

    features = load_daily_features_frame(
        "SPY",
        proc_root=Path("/tmp/daily_features"),
        columns=["iv_dlt25_30d", "iv_dlt75_30d"],
        start="2024-01-03",
        end="2024-01-03",
    )

    assert seen["ticker"] == "SPY"
    assert seen["proc_root"] == Path("/tmp/daily_features")
    assert seen["columns"] == ("trade_date", "iv_dlt25_30d", "iv_dlt75_30d")
    assert list(features.index) == list(pd.to_datetime(["2024-01-03"]))
    assert list(features.columns) == ["iv_dlt25_30d", "iv_dlt75_30d"]
    assert features.iloc[0]["iv_dlt25_30d"] == pytest.approx(0.31)


def test_load_daily_features_frame_raises_on_empty_window(monkeypatch) -> None:
    monkeypatch.setattr(
        data_loading,
        "read_daily_features",
        lambda ticker, *, proc_root, columns=None: pl.DataFrame(
            {
                "trade_date": ["2024-01-02"],
                "iv_dlt25_30d": [0.30],
            }
        ),
    )

    with pytest.raises(ValueError, match="No daily-features rows for SPY"):
        load_daily_features_frame(
            "SPY",
            start="2024-01-03",
            end="2024-01-03",
        )


def test_load_yfinance_close_series_accepts_source_root(
    monkeypatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "yfinance"
    time_series_root = source_root / "time_series"
    time_series_root.mkdir(parents=True)
    (time_series_root / "yfinance_time_series.parquet").touch()
    seen: dict[str, Path] = {}

    def fake_read_yfinance_time_series(*, proc_root, columns=None, tickers=None):
        seen["proc_root"] = Path(proc_root)
        assert tickers == ["SPY"]
        return pl.DataFrame(
            {
                "date": ["2024-01-03", "2024-01-02"],
                "close": [101.0, 100.0],
            }
        )

    monkeypatch.setattr(
        data_loading,
        "read_yfinance_time_series",
        fake_read_yfinance_time_series,
    )

    prices = load_yfinance_close_series("SPY", proc_root=source_root)

    assert seen["proc_root"] == time_series_root
    assert list(prices.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert list(prices) == [100.0, 101.0]


def test_spot_series_from_options_chain_uses_existing_trade_date_index() -> None:
    options = pd.DataFrame(
        {
            "spot_price": [101.0, 101.0, 102.0],
            "strike": [100.0, 105.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
    )
    options.index.name = "trade_date"

    spot = spot_series_from_options_chain(options)

    assert list(spot.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert list(spot) == [101.0, 102.0]
    assert spot.name == "spot_price"
