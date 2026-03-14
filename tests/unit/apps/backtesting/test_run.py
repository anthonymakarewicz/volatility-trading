from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from volatility_trading.backtesting.runner.workflow_types import (
    BacktestDataSourcesSpec,
    BacktestWorkflowSpec,
    NamedStrategyPresetSpec,
    OptionsSourceSpec,
)


def _parse_printed_config(text: str) -> dict[str, Any]:
    return json.loads(text)


def test_help(capsys) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "config-driven backtest workflow" in out
    assert "--ticker" in out
    assert "--output-root" in out


def test_print_config_applies_ticker_override_without_runner_defaults(
    tmp_path: Path,
    capsys,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")
    config_path = tmp_path / "workflow.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  options:",
                "    ticker: SPY",
                "  features:",
                "    ticker: SPY",
                "  hedge:",
                "    ticker: SPY",
                "    provider: yfinance",
                "strategy:",
                "  name: vrp_harvesting",
            ]
        ),
        encoding="utf-8",
    )

    mod.main(
        [
            "--config",
            str(config_path),
            "--ticker",
            "QQQ",
            "--run-id",
            "smoke-run",
            "--print-config",
        ]
    )
    printed = _parse_printed_config(capsys.readouterr().out)

    assert printed["data"]["options"]["ticker"] == "QQQ"
    assert printed["data"]["features"]["ticker"] == "QQQ"
    assert printed["data"]["hedge"]["ticker"] == "QQQ"
    assert "signal" not in printed["strategy"]
    assert printed["reporting"]["run_id"] == "smoke-run"


def test_build_config_does_not_assume_default_strategy() -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")

    config = mod._build_config(mod._parse_args([]))

    assert "strategy" not in config
    assert "rates" not in config["data"]


def test_workflow_payload_requires_explicit_data_rates_for_sourced_financing(
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")
    config_path = tmp_path / "workflow.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  options:",
                "    ticker: SPY",
                "strategy:",
                "  name: vrp_harvesting",
                "broker:",
                "  margin:",
                "    policy:",
                "      cash_rate_source: data_rates",
                "      borrow_rate_spread: 0.02",
            ]
        ),
        encoding="utf-8",
    )

    config = mod._build_config(mod._parse_args(["--config", str(config_path)]))

    with pytest.raises(
        ValueError,
        match="cash_rate_source='data_rates' requires data.rates in the workflow",
    ):
        mod.parse_workflow_config(mod._workflow_config_payload(config))


def test_dry_run_validates_without_executing(monkeypatch, caplog) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")
    workflow = BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(ticker="SPY"),
        ),
        strategy=NamedStrategyPresetSpec(
            name="vrp_harvesting",
            params={},
        ),
    )
    parse_calls: dict[str, Any] = {}
    assemble_calls: dict[str, Any] = {}
    run_calls: dict[str, Any] = {}

    def _parse(config):
        parse_calls["config"] = config
        return workflow

    def _assemble(received_workflow):
        assemble_calls["workflow"] = received_workflow
        return SimpleNamespace(
            strategy=SimpleNamespace(name="vrp_harvesting"),
            benchmark_name=None,
            risk_free_rate=0.0,
        )

    def _run(config):
        run_calls["config"] = config
        raise AssertionError("run_backtest_workflow_config should not be called")

    monkeypatch.setattr(mod, "parse_workflow_config", _parse)
    monkeypatch.setattr(mod, "assemble_workflow_inputs", _assemble)
    monkeypatch.setattr(mod, "run_backtest_workflow_config", _run)
    monkeypatch.setattr(mod, "setup_logging_from_config", lambda _cfg: None)

    caplog.set_level("INFO")
    mod.main(["--dry-run"])

    assert "dry_run" not in parse_calls["config"]
    assert "logging" not in parse_calls["config"]
    assert "strategy" not in parse_calls["config"]
    assert assemble_calls["workflow"] is workflow
    assert "DRY RUN: no actions were executed." in caplog.text
    assert "backtest_run" in caplog.text
    assert run_calls == {}


def test_main_runs_workflow_service_with_merged_config(monkeypatch, caplog) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")
    captured: dict[str, Any] = {}

    def _run(config):
        captured["config"] = config
        return SimpleNamespace(
            workflow=SimpleNamespace(strategy=SimpleNamespace(name="skew_mispricing")),
            trades=pd.DataFrame({"pnl": [1.0, -0.5]}),
            mtm=pd.DataFrame({"delta_pnl": [0.1, 0.2, 0.3]}),
            report_dir=Path("/tmp/backtest-report"),
        )

    monkeypatch.setattr(mod, "run_backtest_workflow_config", _run)
    monkeypatch.setattr(mod, "setup_logging_from_config", lambda _cfg: None)

    caplog.set_level("INFO")
    mod.main(
        [
            "--ticker",
            "IWM",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--output-root",
            "/tmp/reports",
            "--run-id",
            "iwm-2024",
        ]
    )

    config = captured["config"]
    assert "dry_run" not in config
    assert "logging" not in config
    assert "strategy" not in config
    assert config["data"]["options"]["ticker"] == "IWM"
    assert config["run"]["start_date"] == "2024-01-01"
    assert config["run"]["end_date"] == "2024-12-31"
    assert config["reporting"]["output_root"] == "/tmp/reports"
    assert config["reporting"]["run_id"] == "iwm-2024"
    assert "Completed backtest workflow strategy=skew_mispricing" in caplog.text
