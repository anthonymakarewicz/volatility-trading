from pathlib import Path

import pandas as pd

from volatility_trading.backtesting import OptionsMarketData
from volatility_trading.backtesting.data_contracts import OptionsBacktestDataBundle
from volatility_trading.backtesting.runner.assembly import ResolvedWorkflowInputs
from volatility_trading.backtesting.runner.registry import build_strategy_preset
from volatility_trading.backtesting.runner.service import (
    run_backtest_workflow,
    run_backtest_workflow_config,
)
from volatility_trading.backtesting.runner.types import (
    NamedSignalSpec,
    NamedStrategyPresetSpec,
)
from volatility_trading.backtesting.runner.workflow_types import (
    BacktestDataSourcesSpec,
    BacktestWorkflowSpec,
    OptionsSourceSpec,
    ReportingSpec,
    RunWindowSpec,
)


def _sample_workflow(*, save_report_bundle: bool = True) -> BacktestWorkflowSpec:
    return BacktestWorkflowSpec(
        data=BacktestDataSourcesSpec(
            options=OptionsSourceSpec(ticker="SPX"),
        ),
        strategy=NamedStrategyPresetSpec(
            name="vrp_harvesting",
            signal=NamedSignalSpec(name="short_only"),
        ),
        run=RunWindowSpec(start_date="2020-01-01", end_date="2020-01-03"),
        reporting=ReportingSpec(
            output_root=Path("/tmp/runner_service_reports"),
            run_id="runner_service_test",
            save_report_bundle=save_report_bundle,
        ),
    )


def _sample_resolved(workflow: BacktestWorkflowSpec) -> ResolvedWorkflowInputs:
    strategy = build_strategy_preset(workflow.strategy)
    data = OptionsBacktestDataBundle(
        options_market=OptionsMarketData(
            chain=pd.DataFrame(index=pd.to_datetime(["2020-01-01"])),
            symbol="SPX",
        )
    )
    return ResolvedWorkflowInputs(
        workflow=workflow,
        strategy=strategy,
        data=data,
        run_config=workflow.to_backtest_run_config(),
        benchmark=pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            name="SPXTR",
        ),
        benchmark_name="SPXTR",
        risk_free_rate=0.01,
    )


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2020-01-01")],
            "exit_date": [pd.Timestamp("2020-01-03")],
            "contracts": [1],
            "pnl": [12.0],
        }
    )


def _sample_raw_mtm() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {
            "delta_pnl": [0.0, 5.0, 7.0],
            "net_delta": [0.0, 1.0, 1.0],
            "delta": [0.0, 1.0, 1.0],
            "gamma": [0.0, 0.1, 0.1],
            "vega": [0.0, 1.0, 1.0],
            "theta": [0.0, -0.2, -0.2],
            "S": [100.0, 101.0, 102.0],
            "iv": [0.20, 0.21, 0.21],
            "hedge_pnl": [0.0, 0.0, 0.0],
        },
        index=index,
    )


def test_run_backtest_workflow_executes_and_saves_report_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workflow = _sample_workflow(save_report_bundle=True)
    workflow = BacktestWorkflowSpec(
        data=workflow.data,
        strategy=workflow.strategy,
        run=workflow.run,
        reporting=ReportingSpec(
            output_root=tmp_path,
            run_id="runner_service_test",
            save_report_bundle=True,
        ),
        account=workflow.account,
        execution=workflow.execution,
        broker=workflow.broker,
        modeling=workflow.modeling,
    )

    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.assemble_workflow_inputs",
        lambda spec: _sample_resolved(spec),
    )
    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.Backtester.run",
        lambda self: (_sample_trades(), _sample_raw_mtm()),
    )

    result = run_backtest_workflow(workflow)

    assert result.daily_mtm["equity"].iloc[-1] == 100_012.0
    assert result.report_bundle is not None
    assert result.report_dir is not None
    assert (result.report_dir / "manifest.json").exists()
    assert (
        result.report_bundle.run_config["workflow"]["strategy"]["name"]
        == "vrp_harvesting"
    )


def test_run_backtest_workflow_builds_but_does_not_save_report_when_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workflow = _sample_workflow(save_report_bundle=False)
    workflow = BacktestWorkflowSpec(
        data=workflow.data,
        strategy=workflow.strategy,
        run=workflow.run,
        reporting=ReportingSpec(
            output_root=tmp_path,
            run_id="runner_service_no_save",
            save_report_bundle=False,
        ),
        account=workflow.account,
        execution=workflow.execution,
        broker=workflow.broker,
        modeling=workflow.modeling,
    )

    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.assemble_workflow_inputs",
        lambda spec: _sample_resolved(spec),
    )
    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.Backtester.run",
        lambda self: (_sample_trades(), _sample_raw_mtm()),
    )

    result = run_backtest_workflow(workflow)

    assert result.report_bundle is not None
    assert result.report_dir is None
    assert not any(tmp_path.iterdir())


def test_run_backtest_workflow_skips_reporting_when_mtm_is_empty(monkeypatch) -> None:
    workflow = _sample_workflow(save_report_bundle=True)
    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.assemble_workflow_inputs",
        lambda spec: _sample_resolved(spec),
    )
    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.Backtester.run",
        lambda self: (pd.DataFrame(), pd.DataFrame()),
    )

    result = run_backtest_workflow(workflow)

    assert result.daily_mtm.empty
    assert result.report_bundle is None
    assert result.report_dir is None


def test_run_backtest_workflow_config_parses_then_runs(monkeypatch) -> None:
    captured: dict[str, BacktestWorkflowSpec] = {}

    def _fake_run(workflow: BacktestWorkflowSpec):
        captured["workflow"] = workflow
        return "ran"

    monkeypatch.setattr(
        "volatility_trading.backtesting.runner.service.run_backtest_workflow",
        _fake_run,
    )

    result = run_backtest_workflow_config(
        {
            "data": {"options": {"ticker": "SPX"}},
            "strategy": {
                "name": "vrp_harvesting",
                "signal": {"name": "short_only"},
            },
        }
    )

    assert result == "ran"
    assert captured["workflow"].strategy.name == "vrp_harvesting"
    assert captured["workflow"].data.options.ticker == "SPX"
