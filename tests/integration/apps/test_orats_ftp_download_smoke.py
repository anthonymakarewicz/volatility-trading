from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_ftp_download_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_ftp")
    run_help(mod, "Download ORATS FTP ZIP files into a raw directory.")


def test_orats_ftp_download_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_ftp")
    cfg = run_print_config(mod, "config/orats_ftp_download.yml")
    assert_paths_exist(cfg, [("paths", "raw_root")])


def test_orats_ftp_download_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_ftp")

    raw_root = tmp_path / "raw"
    monkeypatch.setattr(mod, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("ORATS_FTP_USER", "user")
    monkeypatch.setenv("ORATS_FTP_PASS", "pass")

    monkeypatch.setattr(
        mod,
        "download",
        lambda **kwargs: pytest.fail("download() should not be called in dry-run"),
    )

    mod.main(
        [
            "--config",
            "config/orats_ftp_download.yml",
            "--raw-root",
            str(raw_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()


def test_orats_ftp_download_dry_run_requires_credentials(monkeypatch) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.download_ftp")
    monkeypatch.setattr(mod, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("ORATS_FTP_USER", raising=False)
    monkeypatch.delenv("ORATS_FTP_PASS", raising=False)
    with pytest.raises(RuntimeError, match="Missing ORATS FTP credentials"):
        mod.main(
            [
                "--config",
                "config/orats_ftp_download.yml",
                "--dry-run",
            ]
        )
