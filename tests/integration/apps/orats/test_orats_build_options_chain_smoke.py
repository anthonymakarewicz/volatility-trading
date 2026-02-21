from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_build_options_chain_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_options_chain")
    run_help(mod, "Build processed ORATS options-chain panels.")


def test_orats_build_options_chain_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_options_chain")
    cfg = run_print_config(mod, "config/orats/options_chain_build.yml")
    assert_paths_exist(
        cfg,
        [
            ("paths", "inter_root"),
            ("paths", "monies_implied_root"),
            ("paths", "proc_root"),
        ],
    )


def test_orats_build_options_chain_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.build_options_chain")

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
            "config/orats/options_chain_build.yml",
            "--inter-strikes-root",
            str(inter_root),
            "--proc-root",
            str(proc_root),
            "--dry-run",
        ]
    )

    assert not proc_root.exists()
