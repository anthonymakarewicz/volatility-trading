from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_build_daily_features_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_daily_features")
    run_help(mod, "Build processed ORATS daily-features panels.")


def test_orats_build_daily_features_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_daily_features")
    cfg = run_print_config(mod, "config/orats_daily_features_build.yml")
    assert_paths_exist(cfg, [("paths", "inter_root"), ("paths", "proc_root")])


def test_orats_build_daily_features_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_daily_features")

    proc_root = tmp_path / "proc"
    inter_root = tmp_path / "inter"

    monkeypatch.setattr(
        mod,
        "build",
        lambda **kwargs: pytest.fail("build() should not be called in dry-run"),
    )

    mod.main(
        [
            "--config",
            "config/orats_daily_features_build.yml",
            "--inter-api-root",
            str(inter_root),
            "--proc-root",
            str(proc_root),
            "--dry-run",
        ]
    )

    assert not proc_root.exists()
