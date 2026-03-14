from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from volatility_trading.backtesting.runner.config_parser import parse_workflow_config


def test_vrp_workflow_config_example_parses() -> None:
    app = importlib.import_module("volatility_trading.apps.backtesting.run")
    config_path = Path("config/backtesting/vrp_harvesting.yml")
    config = app._build_config(app._parse_args(["--config", str(config_path)]))
    workflow_config = app._workflow_config_payload(config)

    workflow = parse_workflow_config(workflow_config)

    assert workflow.strategy.name == "vrp_harvesting"
    assert workflow.strategy.signal is not None
    assert workflow.strategy.signal.name == "short_only"
    assert workflow.data.options.ticker == "SPY"
    assert workflow.broker.margin.model is not None


def test_vrp_workflow_config_example_parses_from_notebook_subdirectory(
    monkeypatch,
) -> None:
    app = importlib.import_module("volatility_trading.apps.backtesting.run")
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root / "notebooks/vrp_harvesting")

    config = app._build_config(
        app._parse_args(["--config", "config/backtesting/vrp_harvesting.yml"])
    )
    workflow = parse_workflow_config(app._workflow_config_payload(config))

    assert workflow.strategy.name == "vrp_harvesting"
    assert (
        workflow.data.options.proc_root
        == repo_root / "data/processed/orats/options_chain"
    )
    assert workflow.reporting.output_root == repo_root / "reports/backtests"


def test_skew_workflow_config_example_parses() -> None:
    app = importlib.import_module("volatility_trading.apps.backtesting.run")
    config_path = Path("config/backtesting/skew_mispricing.yml")
    config = app._build_config(app._parse_args(["--config", str(config_path)]))
    workflow_config = app._workflow_config_payload(config)

    workflow = parse_workflow_config(workflow_config)

    assert workflow.strategy.name == "skew_mispricing"
    assert workflow.strategy.signal is None
    assert workflow.data.options.dte_min == pytest.approx(5.0)
    assert workflow.data.options.dte_max == pytest.approx(60.0)
    assert workflow.data.features is not None
    assert workflow.data.features.ticker == "SPY"
    assert workflow.broker.margin.model is not None
    assert workflow.broker.margin.policy is None
    assert workflow.margin_policy_spec is not None
    assert workflow.margin_policy_spec.maintenance_margin_ratio == pytest.approx(0.80)
    assert workflow.margin_policy_spec.margin_call_grace_days == 2
    assert workflow.margin_policy_spec.liquidation_mode == "target"
    assert workflow.margin_policy_spec.cash_rate_source == "data_rates"
    assert workflow.margin_policy_spec.borrow_rate_spread == pytest.approx(0.02)
