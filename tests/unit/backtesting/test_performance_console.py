import pandas as pd

from volatility_trading.backtesting.performance import (
    format_performance_report,
    print_performance_report,
    print_stressed_risk_metrics,
)
from volatility_trading.backtesting.performance.calculators import (
    compute_performance_metrics,
)


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    trades = pd.DataFrame({"contracts": [1, 2], "pnl": [10.0, -5.0]})
    mtm_daily = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 99.0],
            "delta_pnl": [0.0, 10.0, -11.0],
        },
        index=index,
    )
    return trades, mtm_daily


def test_format_performance_report_contains_expected_sections():
    trades, mtm_daily = _sample_inputs()
    metrics = compute_performance_metrics(trades=trades, mtm_daily=mtm_daily)

    report = format_performance_report(metrics)

    assert "Overall Performance Metrics" in report
    assert "Sharpe Ratio" in report
    assert "Total Trades" in report


def test_print_performance_report_prints_and_returns_metrics(capsys):
    trades, mtm_daily = _sample_inputs()

    metrics = print_performance_report(trades=trades, mtm_daily=mtm_daily)
    captured = capsys.readouterr().out

    assert metrics.trades.total_trades == 2
    assert "Performance by Contract Size" in captured


def test_print_stressed_risk_metrics_prints_and_returns_metrics(capsys):
    _, mtm_daily = _sample_inputs()
    stressed_mtm = pd.DataFrame(
        {"PnL_down_5": [0.0, -2.0, -1.0], "PnL_up_5": [0.0, 1.5, 0.8]},
        index=mtm_daily.index,
    )

    metrics = print_stressed_risk_metrics(
        stressed_mtm=stressed_mtm,
        mtm_daily=mtm_daily,
    )
    captured = capsys.readouterr().out

    assert "Base VaR" in captured
    assert "Stress CVaR" in captured
    assert metrics["base_var"] is not None
    assert metrics["stress_var"] is not None
