from pathlib import Path

import pandas as pd
import pytest

from volatility_trading.backtesting.options_engine.lifecycle import (
    FixedBpsHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
)
from volatility_trading.backtesting.runner import parse_workflow_config


def test_parse_workflow_config_builds_minimal_vrp_workflow() -> None:
    workflow = parse_workflow_config(
        {
            "data": {
                "options": {
                    "ticker": "spx",
                }
            },
            "strategy": {
                "name": "vrp_harvesting",
                "signal": {"name": "short_only"},
            },
        }
    )

    assert workflow.data.options.ticker == "SPX"
    assert workflow.strategy.name == "vrp_harvesting"
    assert workflow.strategy.signal is not None
    assert workflow.strategy.signal.name == "short_only"
    assert workflow.reporting.include_dashboard_plot is True


def test_parse_workflow_config_builds_skew_workflow_with_defaults_and_execution() -> (
    None
):
    workflow = parse_workflow_config(
        {
            "data": {
                "options": {
                    "ticker": "spx",
                    "adapter_name": "canonical",
                    "default_contract_multiplier": 100,
                },
                "features": {
                    "ticker": "spx",
                },
                "hedge": {
                    "ticker": "spy",
                    "price_column": "adj_close",
                    "contract_multiplier": 1.0,
                },
                "benchmark": {
                    "ticker": "iwm",
                },
                "rates": {
                    "provider": "fred",
                    "series_id": "DGS3MO",
                    "column": "DGS3MO",
                },
            },
            "strategy": {
                "name": "skew_mispricing",
                "params": {
                    "target_dte": 30,
                    "delta_target_abs": 0.25,
                },
            },
            "account": {
                "initial_capital": 250_000,
            },
            "execution": {
                "option": {
                    "model": "mid_no_cost",
                },
                "hedge": {
                    "model": "fixed_bps",
                    "params": {"fee_bps": 0.0},
                },
            },
            "run": {
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            "reporting": {
                "output_root": "/tmp/reports",
                "benchmark_name": "IWM TR",
                "include_component_plots": True,
            },
        }
    )

    assert workflow.data.options.ticker == "SPX"
    assert workflow.data.features is not None
    assert workflow.data.features.ticker == "SPX"
    assert workflow.data.hedge is not None
    assert workflow.data.hedge.ticker == "SPY"
    assert workflow.data.hedge.price_column == "adj_close"
    assert workflow.data.benchmark is not None
    assert workflow.data.benchmark.ticker == "IWM"
    assert workflow.data.rates is not None
    assert workflow.data.rates.provider == "fred"
    assert workflow.strategy.name == "skew_mispricing"
    assert workflow.strategy.signal is None
    assert isinstance(
        workflow.execution.option_execution_model,
        MidNoCostOptionExecutionModel,
    )
    assert isinstance(
        workflow.execution.hedge_execution_model,
        FixedBpsHedgeExecutionModel,
    )
    assert workflow.account.initial_capital == pytest.approx(250_000.0)
    assert workflow.reporting.output_root == Path("/tmp/reports")
    assert workflow.reporting.benchmark_name == "IWM TR"
    assert workflow.run.start_date == pd.Timestamp("2020-01-01")
    assert workflow.run.end_date == pd.Timestamp("2020-12-31")


def test_parse_workflow_config_rejects_missing_required_sections() -> None:
    with pytest.raises(
        ValueError,
        match="workflow config is missing required keys: data, strategy",
    ):
        parse_workflow_config({})


def test_parse_workflow_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(
        ValueError,
        match="workflow config contains unsupported keys: extra",
    ):
        parse_workflow_config(
            {
                "data": {"options": {"ticker": "SPX"}},
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {"name": "short_only"},
                },
                "extra": {},
            }
        )


def test_parse_workflow_config_rejects_unknown_execution_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown execution.option model 'unknown'",
    ):
        parse_workflow_config(
            {
                "data": {"options": {"ticker": "SPX"}},
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {"name": "short_only"},
                },
                "execution": {
                    "option": {"model": "unknown"},
                },
            }
        )


def test_parse_workflow_config_rejects_invalid_signal_params() -> None:
    with pytest.raises(
        ValueError,
        match="strategy.signal.params must be a mapping",
    ):
        parse_workflow_config(
            {
                "data": {"options": {"ticker": "SPX"}},
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {
                        "name": "short_only",
                        "params": ["not", "a", "mapping"],
                    },
                },
            }
        )


def test_parse_workflow_config_rejects_non_empty_broker_section_for_now() -> None:
    with pytest.raises(
        ValueError,
        match="broker contains unsupported keys: margin",
    ):
        parse_workflow_config(
            {
                "data": {"options": {"ticker": "SPX"}},
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {"name": "short_only"},
                },
                "broker": {"margin": {}},
            }
        )
