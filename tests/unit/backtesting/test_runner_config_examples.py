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


def test_skew_workflow_config_example_parses() -> None:
    app = importlib.import_module("volatility_trading.apps.backtesting.run")
    config_path = Path("config/backtesting/skew_mispricing.yml")
    config = app._build_config(app._parse_args(["--config", str(config_path)]))
    workflow_config = app._workflow_config_payload(config)

    workflow = parse_workflow_config(workflow_config)

    assert workflow.strategy.name == "skew_mispricing"
    assert workflow.strategy.signal is None
    assert workflow.data.features is not None
    assert workflow.data.features.ticker == "SPY"
    assert workflow.broker.margin.model is not None
    assert workflow.broker.margin.policy is not None
    assert workflow.broker.margin.policy.maintenance_margin_ratio == pytest.approx(0.80)
    assert workflow.broker.margin.policy.margin_call_grace_days == 2
    assert workflow.broker.margin.policy.liquidation_mode == "target"
