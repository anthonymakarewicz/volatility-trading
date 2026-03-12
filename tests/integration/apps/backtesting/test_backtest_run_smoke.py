from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from volatility_trading.datasets.options_chain import options_chain_path

pytestmark = pytest.mark.integration


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _write_minimal_vrp_options(proc_root: Path) -> None:
    _write_parquet(
        options_chain_path(proc_root, "SPY"),
        pl.DataFrame(
            {
                "ticker": ["SPY", "SPY", "SPY"],
                "trade_date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    pd.Timestamp("2020-01-03"),
                ],
                "expiry_date": [
                    pd.Timestamp("2020-01-31"),
                    pd.Timestamp("2020-01-31"),
                    pd.Timestamp("2020-01-31"),
                ],
                "strike": [100.0, 100.0, 100.0],
                "dte": [30, 29, 28],
                "spot_price": [100.0, 101.0, 102.0],
                "call_bid_price": [5.0, 5.8, 6.2],
                "call_ask_price": [5.2, 6.2, 6.5],
                "call_delta": [0.5, 0.5, 0.5],
                "call_gamma": [0.01, 0.01, 0.01],
                "call_vega": [0.10, 0.10, 0.10],
                "call_theta": [-0.02, -0.02, -0.02],
                "call_market_iv": [0.20, 0.22, 0.23],
                "put_bid_price": [5.0, 5.8, 6.2],
                "put_ask_price": [5.2, 6.2, 6.5],
                "put_delta": [-0.5, -0.5, -0.5],
                "put_gamma": [0.01, 0.01, 0.01],
                "put_vega": [0.10, 0.10, 0.10],
                "put_theta": [-0.02, -0.02, -0.02],
                "put_market_iv": [0.20, 0.22, 0.23],
            }
        ),
    )


def test_backtest_run_executes_full_workflow_and_writes_report_bundle(
    tmp_path: Path,
) -> None:
    mod = importlib.import_module("volatility_trading.apps.backtesting.run")

    options_root = tmp_path / "options"
    reports_root = tmp_path / "reports"
    config_path = tmp_path / "workflow.yml"
    report_dir = reports_root / "vrp_harvesting" / "spy_smoke"

    _write_minimal_vrp_options(options_root)
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  options:",
                "    ticker: SPY",
                "    provider: orats",
                f"    proc_root: {options_root}",
                "  rates:",
                "    provider: constant",
                "    constant_rate: 0.0",
                "strategy:",
                "  name: vrp_harvesting",
                "  signal:",
                "    name: short_only",
                "    params: {}",
                "  params:",
                "    target_dte: 30",
                "    max_dte_diff: 7",
                "    rebalance_period: 2",
                "    allow_same_day_reentry_on_rebalance: false",
                "    allow_same_day_reentry_on_max_holding: false",
                "account:",
                "  initial_capital: 100000.0",
                "execution:",
                "  option:",
                "    model: bid_ask_fee",
                "    params:",
                "      commission_per_leg: 0.0",
                "  hedge:",
                "    model: fixed_bps",
                "    params:",
                "      fee_bps: 0.0",
                "broker: {}",
                "modeling: {}",
                "run:",
                "  start_date: '2020-01-01'",
                "  end_date: '2020-01-03'",
                "reporting:",
                f"  output_root: {reports_root}",
                "  run_id: spy_smoke",
                "  save_report_bundle: true",
                "  include_dashboard_plot: false",
                "  include_component_plots: false",
            ]
        ),
        encoding="utf-8",
    )

    mod.main(["--config", str(config_path)])

    assert report_dir.exists()
    assert (report_dir / "manifest.json").exists()
    assert (report_dir / "run_config.json").exists()
    assert (report_dir / "summary_metrics.json").exists()
    assert (report_dir / "trades.csv").exists()
    assert (report_dir / "exposures_daily.csv").exists()

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    run_config = json.loads(
        (report_dir / "run_config.json").read_text(encoding="utf-8")
    )

    assert manifest["metadata"]["strategy_name"] == "vrp_harvesting"
    assert manifest["metadata"]["run_id"] == "spy_smoke"
    assert run_config["config"]["workflow"]["strategy"]["name"] == "vrp_harvesting"
