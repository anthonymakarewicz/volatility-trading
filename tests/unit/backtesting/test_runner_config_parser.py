from pathlib import Path

import pandas as pd
import pytest

from volatility_trading.backtesting.margin import MarginPolicy
from volatility_trading.backtesting.options_engine.lifecycle import (
    FixedBpsHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
)
from volatility_trading.backtesting.runner.config_parser import parse_workflow_config
from volatility_trading.options import RegTMarginModel


def test_parse_workflow_config_applies_vrp_default_signal() -> None:
    workflow = parse_workflow_config(
        {
            "data": {
                "options": {
                    "ticker": "spx",
                }
            },
            "strategy": {
                "name": "vrp_harvesting",
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
                    "dte_min": 5,
                    "dte_max": 60,
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
            "broker": {
                "margin": {
                    "model": {
                        "name": "regt",
                        "params": {"broad_index": False},
                    }
                }
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
    assert workflow.data.options.dte_min == pytest.approx(5.0)
    assert workflow.data.options.dte_max == pytest.approx(60.0)
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
    assert isinstance(workflow.broker.margin.model, RegTMarginModel)
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


def test_parse_workflow_config_rejects_unknown_margin_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown broker.margin.model model 'unknown'",
    ):
        parse_workflow_config(
            {
                "data": {"options": {"ticker": "SPX"}},
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {"name": "short_only"},
                },
                "broker": {
                    "margin": {
                        "model": {"name": "unknown"},
                    }
                },
            }
        )


def test_parse_workflow_config_parses_margin_policy() -> None:
    workflow = parse_workflow_config(
        {
            "data": {"options": {"ticker": "SPX"}},
            "strategy": {
                "name": "vrp_harvesting",
                "signal": {"name": "short_only"},
            },
            "broker": {
                "margin": {
                    "policy": {
                        "apply_financing": True,
                        "cash_rate_annual": 0.01,
                        "borrow_rate_annual": 0.03,
                        "maintenance_margin_ratio": 0.8,
                    }
                }
            },
        }
    )

    policy = workflow.broker.margin.policy

    assert isinstance(policy, MarginPolicy)
    assert isinstance(policy.cash_rate_annual, float)
    assert isinstance(policy.borrow_rate_annual, float)
    assert policy.apply_financing is True
    assert policy.cash_rate_annual == pytest.approx(0.01)
    assert policy.borrow_rate_annual == pytest.approx(0.03)
    assert policy.maintenance_margin_ratio == pytest.approx(0.8)


def test_parse_workflow_config_parses_deferred_margin_policy_spec() -> None:
    workflow = parse_workflow_config(
        {
            "data": {
                "options": {"ticker": "SPX"},
                "rates": {
                    "provider": "fred",
                    "series_id": "DGS3MO",
                },
            },
            "strategy": {
                "name": "vrp_harvesting",
                "signal": {"name": "short_only"},
            },
            "broker": {
                "margin": {
                    "policy": {
                        "apply_financing": True,
                        "cash_rate_source": "data_rates",
                        "borrow_rate_spread": 0.02,
                    }
                }
            },
        }
    )

    assert workflow.margin_policy_spec is not None
    assert workflow.broker.margin.policy is None
    assert workflow.margin_policy_spec.cash_rate_source == "data_rates"
    assert workflow.margin_policy_spec.borrow_rate_spread == pytest.approx(0.02)


def test_parse_workflow_config_rejects_sourced_financing_without_data_rates() -> None:
    with pytest.raises(
        ValueError,
        match="cash_rate_source='data_rates' requires data.rates in the workflow",
    ):
        parse_workflow_config(
            {
                "data": {
                    "options": {"ticker": "SPX"},
                },
                "strategy": {
                    "name": "vrp_harvesting",
                    "signal": {"name": "short_only"},
                },
                "broker": {
                    "margin": {
                        "policy": {
                            "apply_financing": True,
                            "cash_rate_source": "data_rates",
                            "borrow_rate_spread": 0.02,
                        }
                    }
                },
            }
        )
