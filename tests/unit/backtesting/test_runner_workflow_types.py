from pathlib import Path

import pandas as pd
import pytest

from volatility_trading.backtesting import AccountConfig
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
)


def test_run_window_spec_coerces_dates() -> None:
    window = RunWindowSpec(start_date="2020-01-01", end_date="2020-01-31")

    assert window.start_date == pd.Timestamp("2020-01-01")
    assert window.end_date == pd.Timestamp("2020-01-31")


def test_run_window_spec_rejects_invalid_order() -> None:
    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        RunWindowSpec(start_date="2020-02-01", end_date="2020-01-01")


def test_options_source_spec_normalizes_and_validates() -> None:
    spec = OptionsSourceSpec(
        ticker=" spy ",
        adapter_name=" canonical ",
        symbol=" spx ",
        default_contract_multiplier=50.0,
    )

    assert spec.ticker == "SPY"
    assert spec.provider == "orats"
    assert spec.adapter_name == "canonical"
    assert spec.symbol == "spx"
    assert spec.default_contract_multiplier == pytest.approx(50.0)


def test_options_source_spec_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="options provider must be one of orats"):
        OptionsSourceSpec(ticker="SPY", provider="other")


def test_features_source_spec_rejects_blank_ticker() -> None:
    with pytest.raises(ValueError, match="features ticker must be a non-empty string"):
        FeaturesSourceSpec(ticker=" ")


def test_series_source_spec_rejects_invalid_contract_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match="series contract_multiplier must be finite and > 0",
    ):
        SeriesSourceSpec(ticker="SPY", contract_multiplier=0.0)


def test_rates_source_spec_supports_constant_rates() -> None:
    spec = RatesSourceSpec(provider="constant", constant_rate=0.035)

    assert spec.provider == "constant"
    assert spec.constant_rate == pytest.approx(0.035)
    assert spec.series_id is None


def test_rates_source_spec_requires_fred_series_id() -> None:
    with pytest.raises(ValueError, match="fred rates source requires a non-empty"):
        RatesSourceSpec(provider="fred")


def test_reporting_spec_rejects_blank_optional_labels() -> None:
    with pytest.raises(
        ValueError,
        match="reporting benchmark_name must be a non-empty string",
    ):
        ReportingSpec(benchmark_name=" ")


def test_backtest_workflow_spec_maps_to_backtest_run_config() -> None:
    workflow = BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(ticker="SPY", proc_root=Path("/tmp/options")),
            features=FeaturesSourceSpec(
                ticker="SPY",
                proc_root=Path("/tmp/features"),
            ),
            hedge=SeriesSourceSpec(ticker="SPY", price_column="adj_close"),
            benchmark=SeriesSourceSpec(ticker="IWM"),
            rates=RatesSourceSpec(provider="fred", series_id="DGS3MO"),
        ),
        strategy=NamedStrategyPresetSpec(
            name="skew_mispricing",
            signal=NamedSignalSpec(name="zscore", params={"window": 20}),
            params={"target_dte": 30},
        ),
        run=RunWindowSpec(start_date="2020-01-01", end_date="2020-12-31"),
        reporting=ReportingSpec(
            output_root=Path("/tmp/reports"),
            benchmark_name="SPY TR",
            include_component_plots=True,
        ),
        account=AccountConfig(initial_capital=250_000.0),
    )

    run_config = workflow.to_backtest_run_config()

    assert workflow.data.options.proc_root == Path("/tmp/options")
    assert workflow.data.features is not None
    assert workflow.data.features.proc_root == Path("/tmp/features")
    assert workflow.data.hedge is not None
    assert workflow.data.hedge.price_column == "adj_close"
    assert workflow.data.benchmark is not None
    assert workflow.data.rates is not None
    assert workflow.reporting.output_root == Path("/tmp/reports")
    assert workflow.reporting.benchmark_name == "SPY TR"
    assert run_config.account.initial_capital == pytest.approx(250_000.0)
    assert run_config.start_date == pd.Timestamp("2020-01-01")
    assert run_config.end_date == pd.Timestamp("2020-12-31")
