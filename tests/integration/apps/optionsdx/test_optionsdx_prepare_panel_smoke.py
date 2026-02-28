from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_optionsdx_prepare_panel_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.optionsdx.prepare_panel")
    run_help(mod, "Build processed OptionsDX panels from raw archives.")


def test_optionsdx_prepare_panel_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.optionsdx.prepare_panel")
    cfg = run_print_config(mod, "config/optionsdx/prepare_panel.yml")
    assert_paths_exist(cfg, [("paths", "raw_root"), ("paths", "proc_root")])
    assert isinstance(cfg["tickers"], list)
    assert cfg["tickers"]


def test_optionsdx_prepare_panel_cli_overrides_yaml(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.optionsdx.prepare_panel")
    mod.main(
        [
            "--config",
            "config/optionsdx/prepare_panel.yml",
            "--tickers",
            "SPY",
            "QQQ",
            "--start-year",
            "2015",
            "--end-year",
            "2016",
            "--reshape",
            "long",
            "--low-memory",
            "--no-clean",
            "--no-overwrite",
            "--no-verbose",
            "--panel-name",
            "custom_panel.parquet",
            "--print-config",
        ]
    )
    cfg = parse_printed_config(capsys.readouterr().out)
    assert cfg["tickers"] == ["SPY", "QQQ"]
    assert cfg["start_year"] == 2015
    assert cfg["end_year"] == 2016
    assert cfg["reshape"] == "long"
    assert cfg["low_memory"] is True
    assert cfg["do_clean"] is False
    assert cfg["overwrite"] is False
    assert cfg["verbose"] is False
    assert cfg["panel_name"] == "custom_panel.parquet"


def test_optionsdx_prepare_panel_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.optionsdx.prepare_panel")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"

    monkeypatch.setattr(
        mod,
        "prepare_optionsdx_panels",
        lambda **kwargs: pytest.fail(
            "prepare_optionsdx_panels() should not be called in dry-run"
        ),
    )

    mod.main(
        [
            "--config",
            "config/optionsdx/prepare_panel.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()
    assert not proc_root.exists()


def test_optionsdx_prepare_panel_calls_builder_with_expected_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.optionsdx.prepare_panel")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"
    captured: dict[str, object] = {}

    def _fake_prepare(**kwargs):
        captured.update(kwargs)
        return {"SPY": proc_root / "SPY" / "custom_panel.parquet"}

    monkeypatch.setattr(mod, "prepare_optionsdx_panels", _fake_prepare)

    mod.main(
        [
            "--config",
            "config/optionsdx/prepare_panel.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--tickers",
            "SPY",
            "--start-year",
            "2018",
            "--end-year",
            "2019",
            "--reshape",
            "wide",
            "--low-memory",
            "--clean",
            "--overwrite",
            "--panel-name",
            "custom_panel.parquet",
        ]
    )

    assert captured["raw_root"] == raw_root
    assert captured["proc_root"] == proc_root
    assert captured["tickers"] == ["SPY"]
    assert captured["start_year"] == 2018
    assert captured["end_year"] == 2019
    assert captured["reshape"] == "wide"
    assert captured["low_memory"] is True
    assert captured["do_clean"] is True
    assert captured["overwrite"] is True
    assert captured["panel_name"] == "custom_panel.parquet"
