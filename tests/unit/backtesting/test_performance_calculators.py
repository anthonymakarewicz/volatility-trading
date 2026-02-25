import pandas as pd
import pytest

from volatility_trading.backtesting.performance import compute_performance_metrics

# TODO: Create subfolder for performance/


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    trades = pd.DataFrame({"contracts": [1, 1], "pnl": [10.0, -5.0]})
    mtm_daily = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 99.0],
            "delta_pnl": [0.0, 10.0, -11.0],
        },
        index=index,
    )
    return trades, mtm_daily


def test_compute_performance_metrics_returns_expected_values():
    trades, mtm_daily = _sample_inputs()

    metrics = compute_performance_metrics(trades=trades, mtm_daily=mtm_daily)

    assert metrics.returns.total_return == pytest.approx(-0.01)
    assert metrics.drawdown.max_drawdown == pytest.approx(-0.10)
    assert metrics.trades.total_trades == 2
    assert metrics.trades.win_rate == pytest.approx(0.5)
    assert metrics.trades.profit_factor == pytest.approx(2.0)
    assert metrics.trades.total_pnl == pytest.approx(-1.0)


def test_compute_performance_metrics_handles_missing_equity():
    trades = pd.DataFrame({"pnl": [1.0, -1.0]})
    mtm_daily = pd.DataFrame({"delta_pnl": [0.0, 0.5]})

    metrics = compute_performance_metrics(trades=trades, mtm_daily=mtm_daily)

    assert metrics.returns.total_return is None
    assert metrics.drawdown.max_drawdown is None
    assert metrics.trades.total_trades == 2


def test_compute_performance_metrics_accepts_series_risk_free_rate():
    trades, mtm_daily = _sample_inputs()
    rf_series = pd.Series(
        [0.01, 0.02, 0.03],
        index=mtm_daily.index,
    )

    metrics_series = compute_performance_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=rf_series,
    )
    metrics_const = compute_performance_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=0.0,
    )

    assert metrics_series.returns.sharpe is not None
    assert metrics_const.returns.sharpe is not None
    assert metrics_series.returns.sharpe < metrics_const.returns.sharpe
