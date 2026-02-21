from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_api_download_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    run_help(mod, "Download ORATS API endpoint snapshots.")


def test_orats_api_download_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    cfg = run_print_config(mod, "config/orats/api_download.yml")
    assert_paths_exist(cfg, [("paths", "raw_root")])
    assert cfg["endpoint"] == "ivrank"


def test_orats_api_download_cli_overrides_yaml(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    mod.main(
        [
            "--config",
            "config/orats/api_download.yml",
            "--endpoint",
            "monies_implied",
            "--tickers",
            "SPX",
            "AAPL",
            "--print-config",
        ]
    )
    cfg = parse_printed_config(capsys.readouterr().out)
    assert cfg["endpoint"] == "monies_implied"
    assert cfg["tickers"] == ["SPX", "AAPL"]


def test_orats_api_download_dry_run_does_not_write_or_call_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")

    raw_root = tmp_path / "raw"
    monkeypatch.setenv("ORATS_API_KEY", "dummy")
    monkeypatch.setattr(
        mod,
        "download",
        lambda **kwargs: pytest.fail("download() should not be called in dry-run"),
    )

    mod.main(
        [
            "--config",
            "config/orats/api_download.yml",
            "--raw-root",
            str(raw_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()


def test_orats_api_download_dry_run_requires_token(monkeypatch) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    monkeypatch.setattr(mod, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing ORATS API token"):
        mod.main(
            [
                "--config",
                "config/orats/api_download.yml",
                "--dry-run",
            ]
        )
