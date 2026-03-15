from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    BidAskFeeOptionExecutionModel,
    BrokerConfig,
    ExecutionConfig,
    FixedBpsHedgeExecutionModel,
    MarginConfig,
    MarginPolicy,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    load_fred_rate_series,
    load_orats_options_chain_for_backtest,
    to_daily_mtm,
)
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.backtesting.runner.service import run_backtest_workflow_config
from volatility_trading.datasets.fred import fred_rates_path
from volatility_trading.datasets.options_chain import options_chain_path
from volatility_trading.options import RegTMarginModel
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy

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


def _write_vrp_options_with_out_of_band_dte(proc_root: Path) -> None:
    _write_parquet(
        options_chain_path(proc_root, "SPY"),
        pl.DataFrame(
            {
                "ticker": ["SPY", "SPY", "SPY", "SPY"],
                "trade_date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    pd.Timestamp("2020-01-03"),
                ],
                "expiry_date": [
                    pd.Timestamp("2020-01-11"),
                    pd.Timestamp("2020-01-31"),
                    pd.Timestamp("2020-01-31"),
                    pd.Timestamp("2020-01-31"),
                ],
                "strike": [100.0, 100.0, 100.0, 100.0],
                "dte": [10, 30, 29, 28],
                "spot_price": [100.0, 100.0, 101.0, 102.0],
                "call_bid_price": [1.0, 5.0, 5.8, 6.2],
                "call_ask_price": [1.1, 5.2, 6.2, 6.5],
                "call_delta": [0.25, 0.5, 0.5, 0.5],
                "call_gamma": [0.01, 0.01, 0.01, 0.01],
                "call_vega": [0.05, 0.10, 0.10, 0.10],
                "call_theta": [-0.01, -0.02, -0.02, -0.02],
                "call_market_iv": [0.15, 0.20, 0.22, 0.23],
                "put_bid_price": [1.0, 5.0, 5.8, 6.2],
                "put_ask_price": [1.1, 5.2, 6.2, 6.5],
                "put_delta": [-0.25, -0.5, -0.5, -0.5],
                "put_gamma": [0.01, 0.01, 0.01, 0.01],
                "put_vega": [0.05, 0.10, 0.10, 0.10],
                "put_theta": [-0.01, -0.02, -0.02, -0.02],
                "put_market_iv": [0.15, 0.20, 0.22, 0.23],
            }
        ),
    )


def _write_minimal_fred_rates(proc_root: Path) -> None:
    _write_parquet(
        fred_rates_path(proc_root / "rates"),
        pl.DataFrame(
            {
                "date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    pd.Timestamp("2020-01-03"),
                ],
                "dgs3mo": [1.5, 1.6, 1.7],
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
    assert (report_dir / "margin_diagnostics_daily.csv").exists()
    assert (report_dir / "rolling_metrics.csv").exists()
    assert (report_dir / "pnl_attribution_daily.csv").exists()
    assert not (report_dir / "benchmark_comparison.json").exists()

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    run_config = json.loads(
        (report_dir / "run_config.json").read_text(encoding="utf-8")
    )

    assert manifest["metadata"]["strategy_name"] == "vrp_harvesting"
    assert manifest["metadata"]["run_id"] == "spy_smoke"
    assert run_config["config"]["workflow"]["strategy"]["name"] == "vrp_harvesting"
    assert "margin_diagnostics_daily.csv" in manifest["artifacts"]
    assert "rolling_metrics.csv" in manifest["artifacts"]
    assert "pnl_attribution_daily.csv" in manifest["artifacts"]


def test_runner_matches_direct_object_vrp_run_with_rate_sourced_financing(
    tmp_path: Path,
) -> None:
    options_root = tmp_path / "options"
    fred_root = tmp_path / "fred"

    _write_vrp_options_with_out_of_band_dte(options_root)
    _write_minimal_fred_rates(fred_root)

    direct_options = load_orats_options_chain_for_backtest(
        "SPY",
        proc_root=options_root,
        dte_min=20,
        dte_max=40,
    )
    direct_rates = load_fred_rate_series("dgs3mo", proc_root=fred_root)
    direct_strategy = make_vrp_strategy(
        VRPHarvestingSpec(
            signal=ShortOnlySignal(),
            target_dte=30,
            max_dte_diff=7,
            rebalance_period=2,
            margin_budget_pct=0.40,
            allow_same_day_reentry_on_rebalance=False,
            allow_same_day_reentry_on_max_holding=False,
        )
    )
    direct_config = BacktestRunConfig(
        account=AccountConfig(initial_capital=100_000.0),
        execution=ExecutionConfig(
            option_execution_model=BidAskFeeOptionExecutionModel(
                commission_per_leg=0.0
            ),
            hedge_execution_model=FixedBpsHedgeExecutionModel(fee_bps=0.0),
        ),
        broker=BrokerConfig(
            margin=MarginConfig(
                model=RegTMarginModel(broad_index=False),
                policy=MarginPolicy(
                    apply_financing=True,
                    cash_rate_annual=direct_rates,
                    borrow_rate_annual=direct_rates + 0.02,
                ),
            )
        ),
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-01-03"),
    )
    direct_backtester = Backtester(
        data=OptionsBacktestDataBundle(
            options_market=OptionsMarketData(
                chain=direct_options,
                symbol="SPY",
                default_contract_multiplier=100.0,
            )
        ),
        strategy=direct_strategy,
        config=direct_config,
    )

    direct_trades, direct_mtm = direct_backtester.run()
    direct_daily_mtm = to_daily_mtm(
        direct_mtm,
        direct_config.account.initial_capital,
    )

    runner_result = run_backtest_workflow_config(
        {
            "data": {
                "options": {
                    "ticker": "SPY",
                    "provider": "orats",
                    "proc_root": str(options_root),
                    "dte_min": 20,
                    "dte_max": 40,
                },
                "rates": {
                    "provider": "fred",
                    "proc_root": str(fred_root),
                    "series_id": "DGS3MO",
                    "column": "dgs3mo",
                },
            },
            "strategy": {
                "name": "vrp_harvesting",
                "signal": {"name": "short_only", "params": {}},
                "params": {
                    "target_dte": 30,
                    "max_dte_diff": 7,
                    "rebalance_period": 2,
                    "margin_budget_pct": 0.40,
                    "allow_same_day_reentry_on_rebalance": False,
                    "allow_same_day_reentry_on_max_holding": False,
                },
            },
            "account": {"initial_capital": 100000.0},
            "execution": {
                "option": {
                    "model": "bid_ask_fee",
                    "params": {"commission_per_leg": 0.0},
                },
                "hedge": {
                    "model": "fixed_bps",
                    "params": {"fee_bps": 0.0},
                },
            },
            "broker": {
                "margin": {
                    "model": {
                        "name": "regt",
                        "params": {"broad_index": False},
                    },
                    "policy": {
                        "apply_financing": True,
                        "cash_rate_source": "data_rates",
                        "borrow_rate_spread": 0.02,
                    },
                }
            },
            "run": {
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
            },
            "reporting": {
                "output_root": str(tmp_path / "reports"),
                "run_id": "runner_parity",
                "save_report_bundle": False,
                "include_dashboard_plot": False,
                "include_component_plots": False,
            },
        }
    )

    assert_frame_equal(
        runner_result.trades.reset_index(drop=True),
        direct_trades.reset_index(drop=True),
        check_dtype=False,
    )
    assert_frame_equal(
        runner_result.mtm.reset_index(drop=False),
        direct_mtm.reset_index(drop=False),
        check_dtype=False,
    )
    assert_frame_equal(
        runner_result.daily_mtm,
        direct_daily_mtm,
        check_dtype=False,
    )
