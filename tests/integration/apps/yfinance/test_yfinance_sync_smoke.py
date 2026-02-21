from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_yfinance_sync_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.yfinance.sync")
    run_help(mod, "Sync yfinance time-series to parquet files.")


def test_yfinance_sync_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.yfinance.sync")
    cfg = run_print_config(mod, "config/yfinance/time_series_sync.yml")
    assert_paths_exist(cfg, [("paths", "raw_root"), ("paths", "proc_root")])
    assert "tickers" in cfg


def test_yfinance_sync_cli_overrides_yaml(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.yfinance.sync")
    mod.main(
        [
            "--config",
            "config/yfinance/time_series_sync.yml",
            "--tickers",
            "SPY",
            "QQQ",
            "--start",
            "2018-01-01",
            "--end",
            "2018-12-31",
            "--interval",
            "1wk",
            "--auto-adjust",
            "--actions",
            "--overwrite",
            "--print-config",
        ]
    )
    cfg = parse_printed_config(capsys.readouterr().out)
    assert cfg["tickers"] == ["SPY", "QQQ"]
    assert cfg["start"] == "2018-01-01"
    assert cfg["end"] == "2018-12-31"
    assert cfg["interval"] == "1wk"
    assert cfg["auto_adjust"] is True
    assert cfg["actions"] is True
    assert cfg["overwrite"] is True


def test_yfinance_sync_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.yfinance.sync")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"

    monkeypatch.setattr(
        mod,
        "sync_yfinance_time_series",
        lambda **kwargs: pytest.fail(
            "sync_yfinance_time_series() should not be called in dry-run"
        ),
    )

    mod.main(
        [
            "--config",
            "config/yfinance/time_series_sync.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()
    assert not proc_root.exists()


def test_yfinance_sync_calls_sync_with_expected_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.yfinance.sync")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"
    captured: dict[str, object] = {}

    def _fake_sync(**kwargs):
        captured.update(kwargs)
        return proc_root / "yfinance_time_series.parquet"

    monkeypatch.setattr(mod, "sync_yfinance_time_series", _fake_sync)

    mod.main(
        [
            "--config",
            "config/yfinance/time_series_sync.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--tickers",
            "SPY",
            "IWM",
            "--start",
            "2016-01-01",
            "--end",
            "2016-12-31",
            "--interval",
            "1d",
            "--auto-adjust",
            "--actions",
            "--overwrite",
        ]
    )

    assert captured["tickers"] == ["SPY", "IWM"]
    assert captured["raw_root"] == raw_root
    assert captured["proc_root"] == proc_root
    assert captured["start"] == "2016-01-01"
    assert captured["end"] == "2016-12-31"
    assert captured["interval"] == "1d"
    assert captured["auto_adjust"] is True
    assert captured["actions"] is True
    assert captured["overwrite"] is True
