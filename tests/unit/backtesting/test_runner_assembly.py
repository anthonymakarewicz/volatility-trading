from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from volatility_trading.backtesting.data_adapters import CanonicalOptionsChainAdapter
from volatility_trading.backtesting.runner import (
    BacktestDataSourcesSpec,
    BacktestWorkflowSpec,
    FeaturesSourceSpec,
    NamedSignalSpec,
    NamedStrategyPresetSpec,
    OptionsSourceSpec,
    RatesSourceSpec,
    ReportingSpec,
    RunWindowSpec,
    SeriesSourceSpec,
    assemble_workflow_inputs,
)
from volatility_trading.datasets.daily_features import daily_features_path
from volatility_trading.datasets.fred import fred_rates_path
from volatility_trading.datasets.options_chain import options_chain_path
from volatility_trading.datasets.yfinance import yfinance_time_series_path


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_assemble_workflow_inputs_loads_sources_and_builds_runtime_context(
    tmp_path: Path,
) -> None:
    options_root = tmp_path / "options"
    features_root = tmp_path / "features"
    yfinance_root = tmp_path / "yfinance"
    fred_root = tmp_path / "fred_rates"

    _write_parquet(
        options_chain_path(options_root, "SPX"),
        pl.DataFrame(
            {
                "trade_date": [pd.Timestamp("2020-01-01")],
                "expiry_date": [pd.Timestamp("2020-02-01")],
                "strike": [100.0],
                "dte": [31],
                "spot_price": [100.0],
                "call_bid_price": [4.8],
                "call_ask_price": [5.0],
                "call_delta": [0.25],
                "put_bid_price": [4.5],
                "put_ask_price": [4.7],
                "put_delta": [-0.25],
            }
        ),
    )
    _write_parquet(
        daily_features_path(features_root, "SPX"),
        pl.DataFrame(
            {
                "trade_date": [pd.Timestamp("2020-01-01")],
                "iv_dlt25_30d": [0.30],
                "iv_dlt75_30d": [0.20],
            }
        ),
    )
    _write_parquet(
        yfinance_time_series_path(yfinance_root),
        pl.DataFrame(
            {
                "date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),
                ],
                "ticker": ["SPY", "IWM", "SPY"],
                "close": [320.0, 160.0, 320.0],
                "adj_close": [319.0, 159.0, 319.0],
            }
        ),
    )
    _write_parquet(
        fred_rates_path(fred_root),
        pl.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")],
                "dgs3mo": [0.015],
            }
        ),
    )

    workflow = BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(
                ticker="SPX",
                proc_root=options_root,
                default_contract_multiplier=100.0,
            ),
            features=FeaturesSourceSpec(
                ticker="SPX",
                proc_root=features_root,
            ),
            hedge=SeriesSourceSpec(
                ticker="SPY",
                proc_root=yfinance_root,
                price_column="adj_close",
            ),
            benchmark=SeriesSourceSpec(
                ticker="IWM",
                proc_root=yfinance_root,
            ),
            rates=RatesSourceSpec(
                provider="fred",
                series_id="DGS3MO",
                proc_root=fred_root,
            ),
        ),
        strategy=NamedStrategyPresetSpec(
            name="skew_mispricing",
            params={"target_dte": 30},
        ),
        run=RunWindowSpec(start_date="2020-01-01", end_date="2020-01-31"),
        reporting=ReportingSpec(benchmark_name="IWM TR"),
    )

    resolved = assemble_workflow_inputs(workflow)

    assert resolved.strategy.name == "skew_mispricing"
    assert isinstance(
        resolved.data.options_market.options_adapter,
        CanonicalOptionsChainAdapter,
    )
    assert resolved.data.option_symbol == "SPX"
    assert resolved.data.option_contract_multiplier == pytest.approx(100.0)
    assert resolved.data.features is not None
    assert "iv_dlt25_30d" in resolved.data.features.columns
    assert resolved.data.hedge_market is not None
    assert resolved.data.hedge_market.mid.iloc[0] == pytest.approx(319.0)
    assert resolved.benchmark is not None
    assert resolved.benchmark.iloc[0] == pytest.approx(160.0)
    assert isinstance(resolved.risk_free_rate, pd.Series)
    assert resolved.risk_free_rate.iloc[0] == pytest.approx(0.015)
    assert resolved.benchmark_name == "IWM TR"
    assert resolved.run_config.start_date == pd.Timestamp("2020-01-01")
    assert resolved.run_config.end_date == pd.Timestamp("2020-01-31")


def test_assemble_workflow_inputs_rejects_unknown_options_adapter_name() -> None:
    workflow = BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(
                ticker="SPX",
                adapter_name="unknown",
            ),
        ),
        strategy=NamedStrategyPresetSpec(
            name="vrp_harvesting",
            signal=NamedSignalSpec(name="short_only"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unknown options adapter_name 'unknown'",
    ):
        assemble_workflow_inputs(workflow)


def test_assemble_workflow_inputs_resolves_constant_rate_input() -> None:
    options_root = Path("/tmp/runner_assembly_constant_rate_options")
    _write_parquet(
        options_chain_path(options_root, "SPX"),
        pl.DataFrame(
            {
                "trade_date": [pd.Timestamp("2020-01-01")],
                "expiry_date": [pd.Timestamp("2020-02-01")],
                "strike": [100.0],
                "dte": [31],
                "spot_price": [100.0],
                "call_bid_price": [4.8],
                "call_ask_price": [5.0],
                "call_delta": [0.25],
                "put_bid_price": [4.5],
                "put_ask_price": [4.7],
                "put_delta": [-0.25],
            }
        ),
    )
    workflow = BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(ticker="SPX", proc_root=options_root),
            rates=RatesSourceSpec(provider="constant", constant_rate=0.02),
        ),
        strategy=NamedStrategyPresetSpec(
            name="vrp_harvesting",
            signal=NamedSignalSpec(name="short_only"),
        ),
    )

    resolved = assemble_workflow_inputs(workflow)

    assert isinstance(resolved.risk_free_rate, float)
    assert resolved.risk_free_rate == pytest.approx(0.02)
