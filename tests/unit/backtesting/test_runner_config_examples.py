from __future__ import annotations

import importlib
from pathlib import Path

from volatility_trading.backtesting.runner import parse_workflow_config


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
