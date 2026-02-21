from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_ftp_extract_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.extract_ftp")
    run_help(mod, "Extract ORATS FTP ZIPs into intermediate Parquet.")


def test_orats_ftp_extract_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.extract_ftp")
    cfg = run_print_config(mod, "config/orats/ftp_extract.yml")
    assert_paths_exist(cfg, [("paths", "raw_root"), ("paths", "inter_root")])


def test_orats_ftp_extract_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.extract_ftp")

    raw_root = tmp_path / "raw"
    inter_root = tmp_path / "inter"

    monkeypatch.setattr(
        mod,
        "extract",
        lambda **kwargs: pytest.fail("extract() should not be called in dry-run"),
    )

    mod.main(
        [
            "--config",
            "config/orats/ftp_extract.yml",
            "--raw-root",
            str(raw_root),
            "--out-root",
            str(inter_root),
            "--dry-run",
        ]
    )

    assert not raw_root.exists()
    assert not inter_root.exists()
