from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_orats_qc_options_chain_help_exits_cleanly(run_help) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_options_chain")
    run_help(mod, "Run QC on ORATS options-chain panels.")


def test_orats_qc_options_chain_print_config_outputs_json(
    run_print_config,
    assert_paths_exist,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_options_chain")
    cfg = run_print_config(mod, "config/orats_qc_options_chain.yml")
    assert_paths_exist(cfg, [("paths", "proc_root")])


def test_orats_qc_options_chain_dry_run_no_side_effects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.orats.qc_options_chain")
    proc_root = tmp_path / "proc"

    monkeypatch.setattr(
        mod,
        "run_options_chain_qc",
        lambda **kwargs: pytest.fail("run_options_chain_qc() should not be called"),
    )

    mod.main(
        [
            "--config",
            "config/orats_qc_options_chain.yml",
            "--proc-root",
            str(proc_root),
            "--tickers",
            "SPX",
            "--dry-run",
        ]
    )
