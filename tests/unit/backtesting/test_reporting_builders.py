import pandas as pd
import pytest

from volatility_trading.backtesting.reporting.builders import (
    build_equity_and_drawdown_table,
    build_exposures_daily_table,
    build_summary_metrics,
)


def _sample_mtm_daily() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {
            "equity": [100.0, 110.0, 99.0],
            "delta": [1.0, 2.0, 3.0],
            "net_delta": [1.5, 2.5, 3.5],
            "gamma": [0.1, 0.2, 0.3],
            "vega": [10.0, 11.0, 12.0],
            "theta": [-1.0, -1.2, -1.1],
            "hedge_pnl": [0.0, 0.5, -0.2],
        },
        index=index,
    )


def test_build_summary_metrics_returns_expected_core_values():
    trades = pd.DataFrame({"pnl": [10.0, -5.0]})
    mtm_daily = _sample_mtm_daily()

    summary = build_summary_metrics(trades=trades, mtm_daily=mtm_daily)

    assert summary.total_trades == 2
    assert summary.total_return == pytest.approx(-0.01)
    assert summary.max_drawdown == pytest.approx(-0.10)
    assert summary.win_rate == pytest.approx(0.5)
    assert summary.profit_factor == pytest.approx(2.0)


def test_build_summary_metrics_accepts_series_risk_free_rate():
    trades = pd.DataFrame({"pnl": [10.0, -5.0]})
    mtm_daily = _sample_mtm_daily()
    rf_series = pd.Series([0.01, 0.01, 0.02], index=mtm_daily.index)

    summary = build_summary_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=rf_series,
    )

    assert summary.sharpe is not None


def test_build_equity_and_drawdown_table_includes_benchmark_columns():
    mtm_daily = _sample_mtm_daily()
    benchmark = pd.Series(
        [3000.0, 3030.0, 3060.0], index=mtm_daily.index, name="benchmark"
    )

    out = build_equity_and_drawdown_table(mtm_daily=mtm_daily, benchmark=benchmark)

    assert "benchmark_rebased" in out.columns
    assert "benchmark_drawdown" in out.columns
    assert out.index.equals(mtm_daily.index)
    assert out["benchmark_rebased"].iloc[0] == pytest.approx(
        mtm_daily["equity"].iloc[0]
    )


def test_build_exposures_daily_table_selects_expected_columns():
    mtm_daily = _sample_mtm_daily()

    out = build_exposures_daily_table(mtm_daily)

    assert list(out.columns) == [
        "delta",
        "net_delta",
        "gamma",
        "vega",
        "theta",
        "hedge_pnl",
    ]
    assert len(out) == len(mtm_daily)
