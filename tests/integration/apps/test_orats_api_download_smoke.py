from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.integration
def test_orats_api_download_help_exits_cleanly(capsys) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Download ORATS API endpoint snapshots" in out


@pytest.mark.integration
def test_orats_api_download_print_config_outputs_json(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    mod.main(
        [
            "--config",
            "config/orats_api_download.yml",
            "--print-config",
        ]
    )
    cfg = parse_printed_config(capsys.readouterr().out)
    assert "paths" in cfg
    assert "raw_root" in cfg["paths"]
    assert cfg["endpoint"] == "ivrank"


@pytest.mark.integration
def test_orats_api_download_cli_overrides_yaml(
    capsys,
    parse_printed_config,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    mod.main(
        [
            "--config",
            "config/orats_api_download.yml",
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


@pytest.mark.integration
def test_orats_api_download_dry_run_does_not_write_or_call_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")

    raw_root = tmp_path / "raw"
    monkeypatch.setenv("ORATS_API_KEY", "dummy")

    called = {"download": 0}

    def _download(**kwargs):
        called["download"] += 1
        raise AssertionError("download() should not be called during --dry-run")

    monkeypatch.setattr(mod, "download", _download)

    mod.main(
        [
            "--config",
            "config/orats_api_download.yml",
            "--raw-root",
            str(raw_root),
            "--dry-run",
        ]
    )

    assert called["download"] == 0
    assert not raw_root.exists()


@pytest.mark.integration
def test_orats_api_download_dry_run_requires_token(monkeypatch) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_api")
    monkeypatch.setattr(mod, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing ORATS API token"):
        mod.main(
            [
                "--config",
                "config/orats_api_download.yml",
                "--dry-run",
            ]
        )
