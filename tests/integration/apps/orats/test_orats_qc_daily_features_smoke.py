from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_qc_daily_features_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_daily_features")
    run_help(mod, "Run QC on ORATS daily-features panels.")


def test_orats_qc_daily_features_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_daily_features")
    cfg = run_print_config(mod, "config/orats/qc_daily_features.yml")
    assert_paths_exist(cfg, [("paths", "proc_root")])


def test_orats_qc_daily_features_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_daily_features")
    proc_root = tmp_path / "proc"

    monkeypatch.setattr(
        mod,
        "run_daily_features_qc",
        lambda **kwargs: pytest.fail("run_daily_features_qc() should not be called"),
    )

    mod.main(
        [
            "--config",
            "config/orats/qc_daily_features.yml",
            "--proc-root",
            str(proc_root),
            "--tickers",
            "SPX",
            "--dry-run",
        ]
    )
