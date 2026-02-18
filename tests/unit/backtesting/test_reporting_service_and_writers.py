import json
import re

import pandas as pd

from volatility_trading.backtesting.reporting import (
    build_backtest_report_bundle,
    create_run_id,
    save_backtest_report_bundle,
)


def _sample_inputs():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    trades = pd.DataFrame(
        {
            "entry_date": [index[0]],
            "exit_date": [index[-1]],
            "contracts": [2],
            "pnl": [12.5],
        }
    )
    mtm_daily = pd.DataFrame(
        {
            "equity": [100_000.0, 100_200.0, 100_150.0],
            "delta": [0.0, 1.0, -1.0],
            "net_delta": [0.0, 1.0, -1.0],
            "gamma": [0.1, 0.2, 0.1],
            "vega": [5.0, 5.5, 4.8],
            "theta": [-2.0, -2.1, -2.2],
            "hedge_pnl": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    benchmark = pd.Series([3000.0, 3010.0, 3020.0], index=index)
    return trades, mtm_daily, benchmark


def test_create_run_id_uses_expected_utc_format():
    run_id = create_run_id()
    assert re.fullmatch(r"\d{8}_\d{6}", run_id)


def test_build_and_save_report_bundle_writes_core_files(tmp_path):
    trades, mtm_daily, benchmark = _sample_inputs()
    bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=mtm_daily,
        benchmark=benchmark,
        strategy_name="vrp_harvesting",
        benchmark_name="SP500TR",
        run_config={"holding_period": 10, "target_dte": 30},
        run_id="unit_test_run",
        include_dashboard_plot=False,
    )

    run_dir = save_backtest_report_bundle(bundle, output_root=tmp_path)

    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "summary_metrics.json").exists()
    assert (run_dir / "equity_and_drawdown.csv").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "exposures_daily.csv").exists()
    assert (run_dir / "manifest.json").exists()

    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["run_id"] == "unit_test_run"
    assert payload["config"]["target_dte"] == 30


def test_save_report_bundle_writes_dashboard_plot_when_enabled(tmp_path):
    trades, mtm_daily, benchmark = _sample_inputs()
    bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=mtm_daily,
        benchmark=benchmark,
        strategy_name="vrp_harvesting",
        benchmark_name="SP500TR",
        run_config={"holding_period": 10},
        run_id="unit_test_plot_run",
        include_dashboard_plot=True,
    )

    run_dir = save_backtest_report_bundle(bundle, output_root=tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "performance_dashboard.png" in manifest["artifacts"]["plots"]
    assert (run_dir / "plots" / "performance_dashboard.png").exists()


def test_save_report_bundle_writes_component_plots_when_enabled(tmp_path):
    trades, mtm_daily, benchmark = _sample_inputs()
    bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=mtm_daily,
        benchmark=benchmark,
        strategy_name="vrp_harvesting",
        benchmark_name="SP500TR",
        run_config={"holding_period": 10},
        run_id="unit_test_components_plot_run",
        include_dashboard_plot=False,
        include_component_plots=True,
    )

    run_dir = save_backtest_report_bundle(bundle, output_root=tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "equity_vs_benchmark.png" in manifest["artifacts"]["plots"]
    assert "drawdown.png" in manifest["artifacts"]["plots"]
    assert "greeks_exposure.png" in manifest["artifacts"]["plots"]
    assert (run_dir / "plots" / "equity_vs_benchmark.png").exists()
    assert (run_dir / "plots" / "drawdown.png").exists()
    assert (run_dir / "plots" / "greeks_exposure.png").exists()
