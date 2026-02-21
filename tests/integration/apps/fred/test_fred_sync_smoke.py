from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_fred_sync_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.fred.sync")
    run_help(mod, "Sync FRED domains to parquet files.")


def test_fred_sync_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.fred.sync")
    cfg = run_print_config(mod, "config/fred/sync.yml")
    assert_paths_exist(
        cfg,
        [
            ("paths", "raw_root"),
            ("paths", "proc_root"),
            ("fred", "token_env"),
            ("domains", "rates"),
            ("domains", "market"),
        ],
    )


def test_fred_sync_cli_overrides_yaml(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.fred.sync")
    mod.main(
        [
            "--config",
            "config/fred/sync.yml",
            "--domains",
            "rates",
            "--start",
            "2015-01-01",
            "--end",
            "2015-12-31",
            "--token-env",
            "FRED_TEST_API_KEY",
            "--no-business-days",
            "--overwrite",
            "--print-config",
        ]
    )
    cfg = parse_printed_config(capsys.readouterr().out)
    assert cfg["domain_names"] == ["rates"]
    assert cfg["start"] == "2015-01-01"
    assert cfg["end"] == "2015-12-31"
    assert cfg["fred"]["token_env"] == "FRED_TEST_API_KEY"
    assert cfg["asfreq_business_days"] is False
    assert cfg["overwrite"] is True


def test_fred_sync_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.fred.sync")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"

    monkeypatch.setattr(
        mod,
        "sync_fred_domains",
        lambda **kwargs: pytest.fail(
            "sync_fred_domains() should not be called in dry-run"
        ),
    )

    mod.main(
        [
            "--config",
            "config/fred/sync.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()
    assert not proc_root.exists()


def test_fred_sync_calls_sync_with_expected_arguments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.fred.sync")

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"
    captured: dict[str, object] = {}

    def _fake_sync(**kwargs):
        captured.update(kwargs)
        return {"rates": proc_root / "rates" / "fred_rates.parquet"}

    monkeypatch.setattr(mod, "sync_fred_domains", _fake_sync)

    mod.main(
        [
            "--config",
            "config/fred/sync.yml",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--domains",
            "rates",
            "--start",
            "2010-01-01",
            "--end",
            "2010-12-31",
            "--token",
            "dummy_token",
            "--token-env",
            "FRED_TEST_API_KEY",
            "--no-business-days",
            "--overwrite",
        ]
    )

    assert captured["raw_root"] == raw_root
    assert captured["proc_root"] == proc_root
    assert captured["domains"] == {
        "rates": {
            "dgs3mo": "DGS3MO",
            "dgs2": "DGS2",
            "dgs10": "DGS10",
        }
    }
    assert captured["start"] == "2010-01-01"
    assert captured["end"] == "2010-12-31"
    assert captured["token"] == "dummy_token"
    assert captured["token_env"] == "FRED_TEST_API_KEY"
    assert captured["asfreq_business_days"] is False
    assert captured["overwrite"] is True
