import json
import re

import numpy as np
import pandas as pd

from volatility_trading.backtesting.reporting import (
    build_backtest_report_bundle,
    create_run_id,
    save_backtest_report_bundle,
)


def _sample_inputs():
    index = pd.date_range("2020-01-01", periods=70, freq="D")
    trades = pd.DataFrame(
        {
            "entry_date": [index[0]],
            "exit_date": [index[-1]],
            "contracts": [2],
            "pnl": [12.5],
            "risk_per_contract": [250.0],
            "risk_worst_scenario": ["rr.selloff_steepen"],
            "risk_worst_d_spot": [-10.0],
            "risk_worst_d_volatility": [0.05],
            "risk_worst_d_risk_reversal": [-0.05],
            "risk_worst_d_rate": [0.0],
            "risk_worst_dt_years": [0.0],
            "entry_stress_points": [
                [
                    {
                        "scenario_name": "core.selloff_severe",
                        "stress_pnl_per_contract": -100.0,
                        "is_worst_scenario": False,
                        "d_spot": -10.0,
                        "d_volatility": 0.05,
                        "d_risk_reversal": 0.0,
                        "d_rate": 0.0,
                        "dt_years": 0.0,
                    },
                    {
                        "scenario_name": "rr.selloff_steepen",
                        "stress_pnl_per_contract": -250.0,
                        "is_worst_scenario": True,
                        "d_spot": -10.0,
                        "d_volatility": 0.05,
                        "d_risk_reversal": -0.05,
                        "d_rate": 0.0,
                        "dt_years": 0.0,
                    },
                ]
            ],
        }
    )
    equity = 100_000.0 + np.linspace(0.0, 3_000.0, len(index))
    delta_pnl = pd.Series(equity, index=index).diff().fillna(0.0)
    mtm_daily = pd.DataFrame(
        {
            "equity": equity,
            "delta": np.linspace(0.0, 1.0, len(index)),
            "net_delta": np.linspace(0.0, 1.0, len(index)),
            "gamma": np.linspace(0.1, 0.2, len(index)),
            "vega": np.linspace(5.0, 6.0, len(index)),
            "theta": np.linspace(-2.0, -2.3, len(index)),
            "hedge_pnl": np.zeros(len(index)),
            "delta_pnl": delta_pnl,
            "Delta_PnL": delta_pnl * 0.4,
            "Unhedged_Delta_PnL": delta_pnl * 0.5,
            "Gamma_PnL": delta_pnl * 0.1,
            "Vega_PnL": delta_pnl * 0.2,
            "Theta_PnL": delta_pnl * 0.05,
            "Other_PnL": delta_pnl * 0.25,
            "open_contracts": np.full(len(index), 2),
            "margin_per_contract": np.full(len(index), 500.0),
            "initial_margin_requirement": np.full(len(index), 1_000.0),
            "maintenance_margin_requirement": np.full(len(index), 800.0),
            "margin_excess": equity - 800.0,
            "margin_deficit": np.zeros(len(index)),
            "in_margin_call": np.zeros(len(index), dtype=bool),
            "margin_call_days": np.zeros(len(index)),
            "forced_liquidation": np.zeros(len(index), dtype=bool),
            "contracts_liquidated": np.zeros(len(index)),
            "financing_pnl": np.zeros(len(index)),
            "hedge_turnover": np.zeros(len(index)),
            "hedge_trade_count": np.zeros(len(index)),
        },
        index=index,
    )
    benchmark = pd.Series(np.linspace(3000.0, 3200.0, len(index)), index=index)
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
    assert (run_dir / "entry_stress_diagnostics.parquet").exists()
    assert (run_dir / "stress_scenario_summary.csv").exists()
    assert (run_dir / "exposures_daily.csv").exists()
    assert (run_dir / "margin_diagnostics_daily.csv").exists()
    assert (run_dir / "rolling_metrics.csv").exists()
    assert (run_dir / "pnl_attribution_daily.csv").exists()
    assert (run_dir / "benchmark_comparison.json").exists()
    assert (run_dir / "manifest.json").exists()

    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark_payload = json.loads(
        (run_dir / "benchmark_comparison.json").read_text(encoding="utf-8")
    )
    written_trades = pd.read_csv(run_dir / "trades.csv")
    stress = pd.read_parquet(run_dir / "entry_stress_diagnostics.parquet")
    stress_summary = pd.read_csv(run_dir / "stress_scenario_summary.csv")
    assert payload["metadata"]["run_id"] == "unit_test_run"
    assert payload["config"]["target_dte"] == 30
    assert "entry_stress_diagnostics.parquet" in manifest["artifacts"]
    assert "stress_scenario_summary.csv" in manifest["artifacts"]
    assert "margin_diagnostics_daily.csv" in manifest["artifacts"]
    assert "rolling_metrics.csv" in manifest["artifacts"]
    assert "pnl_attribution_daily.csv" in manifest["artifacts"]
    assert benchmark_payload["strategy"]["total_return"] is not None
    assert written_trades.loc[0, "risk_worst_d_risk_reversal"] == -0.05
    assert len(stress) == 2
    assert stress.loc[1, "scenario_name"] == "rr.selloff_steepen"
    assert stress.loc[1, "d_risk_reversal"] == -0.05
    assert len(stress_summary) == 2
    rr_row = stress_summary.loc[
        stress_summary["scenario_name"] == "rr.selloff_steepen"
    ].iloc[0]
    assert rr_row["times_worst"] == 1
    assert rr_row["max_loss_per_contract"] == 250.0


def test_save_report_bundle_omits_benchmark_comparison_without_benchmark(tmp_path):
    trades, mtm_daily, _benchmark = _sample_inputs()
    bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=mtm_daily,
        benchmark=None,
        strategy_name="vrp_harvesting",
        benchmark_name=None,
        run_config={"holding_period": 10},
        run_id="unit_test_no_benchmark",
        include_dashboard_plot=False,
    )

    run_dir = save_backtest_report_bundle(bundle, output_root=tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert bundle.benchmark_comparison is None
    assert not (run_dir / "benchmark_comparison.json").exists()
    assert "benchmark_comparison.json" not in manifest["artifacts"]


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

    expected_plots = {
        "equity_vs_benchmark.png",
        "drawdown.png",
        "greeks_exposure.png",
        "margin_account.png",
        "rolling_metrics.png",
        "pnl_attribution.png",
    }
    assert expected_plots.issubset(set(manifest["artifacts"]["plots"]))
    for plot_name in expected_plots:
        assert (run_dir / "plots" / plot_name).exists()


def test_save_report_bundle_serializes_trade_legs_as_json_for_csv(tmp_path):
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    trades = pd.DataFrame(
        {
            "entry_date": [index[0]],
            "exit_date": [index[-1]],
            "contracts": [1],
            "pnl": [7.5],
            "trade_legs": [
                [
                    {
                        "leg_index": 0,
                        "option_type": "call",
                        "expiry_date": pd.Timestamp("2020-01-31"),
                        "side": -1,
                        "entry_price": 5.2,
                        "exit_price": 4.7,
                    }
                ]
            ],
        }
    )
    mtm_daily = pd.DataFrame(
        {
            "equity": [100_000.0, 100_100.0, 100_120.0],
            "delta_pnl": [0.0, 100.0, 20.0],
        },
        index=index,
    )

    bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=mtm_daily,
        benchmark=None,
        strategy_name="vrp_harvesting",
        benchmark_name=None,
        run_config={"holding_period": 10},
        run_id="unit_test_trade_legs_json",
        include_dashboard_plot=False,
    )

    assert isinstance(bundle.trades.loc[0, "trade_legs"], list)

    run_dir = save_backtest_report_bundle(bundle, output_root=tmp_path)
    written = pd.read_csv(run_dir / "trades.csv")
    payload = json.loads(written.loc[0, "trade_legs"])
    assert isinstance(payload, list)
    assert payload[0]["leg_index"] == 0
    assert payload[0]["expiry_date"] == "2020-01-31T00:00:00"
