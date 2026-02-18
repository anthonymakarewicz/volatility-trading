import matplotlib
import pandas as pd

from volatility_trading.backtesting.reporting import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_greeks_exposure,
    plot_performance_dashboard,
)

matplotlib.use("Agg")


def test_plot_performance_dashboard_returns_figure_with_expected_axes():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    mtm_daily = pd.DataFrame(
        {
            "equity": [100.0, 101.0, 99.5],
            "net_delta": [0.0, 1.0, -1.0],
            "gamma": [0.1, 0.2, 0.15],
            "vega": [5.0, 6.0, 4.0],
            "theta": [-1.0, -0.9, -1.1],
        },
        index=index,
    )
    benchmark = pd.Series([3000.0, 3020.0, 3010.0], index=index)

    fig = plot_performance_dashboard(
        benchmark=benchmark,
        mtm_daily=mtm_daily,
        strategy_name="VRP",
        benchmark_name="SP500TR",
    )

    assert fig is not None
    assert len(fig.axes) == 6


def test_component_plot_builders_return_figures():
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    mtm_daily = pd.DataFrame(
        {
            "equity": [100.0, 101.0, 99.5],
            "net_delta": [0.0, 1.0, -1.0],
            "gamma": [0.1, 0.2, 0.15],
            "vega": [5.0, 6.0, 4.0],
            "theta": [-1.0, -0.9, -1.1],
        },
        index=index,
    )
    benchmark = pd.Series([3000.0, 3020.0, 3010.0], index=index)

    fig_eq = plot_equity_vs_benchmark(benchmark=benchmark, mtm_daily=mtm_daily)
    fig_dd = plot_drawdown(benchmark=benchmark, mtm_daily=mtm_daily)
    fig_gr = plot_greeks_exposure(mtm_daily=mtm_daily)

    assert len(fig_eq.axes) == 1
    assert len(fig_dd.axes) == 1
    assert len(fig_gr.axes) == 4
