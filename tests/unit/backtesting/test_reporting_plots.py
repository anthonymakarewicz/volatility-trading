import matplotlib
import pandas as pd

from volatility_trading.backtesting.reporting.plots import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_full_performance,
    plot_greeks_exposure,
    plot_performance_dashboard,
    plot_pnl_attribution,
    plot_stressed_pnl,
)

matplotlib.use("Agg")


def _sample_mtm_daily() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {
            "equity": [100.0, 101.0, 99.5],
            "net_delta": [0.0, 1.0, -1.0],
            "gamma": [0.1, 0.2, 0.15],
            "vega": [5.0, 6.0, 4.0],
            "theta": [-1.0, -0.9, -1.1],
            "Delta_PnL": [0.0, 0.5, -0.7],
            "Gamma_PnL": [0.0, 0.1, 0.05],
            "Vega_PnL": [0.0, 0.2, -0.3],
            "Theta_PnL": [0.0, -0.1, -0.1],
            "Other_PnL": [0.0, 0.3, -0.2],
        },
        index=index,
    )


def _sample_benchmark(index: pd.Index) -> pd.Series:
    return pd.Series([3000.0, 3020.0, 3010.0], index=index)


def test_plot_performance_dashboard_returns_figure_with_expected_axes():
    mtm_daily = _sample_mtm_daily()
    benchmark = _sample_benchmark(mtm_daily.index)

    fig = plot_performance_dashboard(
        benchmark=benchmark,
        mtm_daily=mtm_daily,
        strategy_name="VRP",
        benchmark_name="SP500TR",
    )

    assert fig is not None
    assert len(fig.axes) == 6


def test_component_plot_builders_return_figures():
    mtm_daily = _sample_mtm_daily()
    benchmark = _sample_benchmark(mtm_daily.index)

    fig_eq = plot_equity_vs_benchmark(benchmark=benchmark, mtm_daily=mtm_daily)
    fig_dd = plot_drawdown(benchmark=benchmark, mtm_daily=mtm_daily)
    fig_gr = plot_greeks_exposure(mtm_daily=mtm_daily)

    assert len(fig_eq.axes) == 1
    assert len(fig_dd.axes) == 1
    assert len(fig_gr.axes) == 4


def test_legacy_performance_and_attribution_plots_return_figures():
    mtm_daily = _sample_mtm_daily()
    benchmark = _sample_benchmark(mtm_daily.index)

    fig_perf = plot_full_performance(benchmark=benchmark, mtm_daily=mtm_daily)
    fig_attr = plot_pnl_attribution(daily_mtm=mtm_daily)

    assert len(fig_perf.axes) == 6
    assert len(fig_attr.axes) == 1


def test_plot_stressed_pnl_returns_figure_for_prefixed_scenario_columns():
    mtm_daily = _sample_mtm_daily()
    stressed_mtm = pd.DataFrame(
        {
            "PnL_down_5": [0.0, -2.0, -1.0],
            "PnL_up_5": [0.0, 1.5, 1.0],
        },
        index=mtm_daily.index,
    )
    scenarios = {"down_5": {"dS_pct": -0.05}, "up_5": {"dS_pct": 0.05}}

    fig = plot_stressed_pnl(
        stressed_mtm=stressed_mtm,
        daily_mtm=mtm_daily,
        scenarios=scenarios,
    )

    assert len(fig.axes) == 1
