import matplotlib
import numpy as np
import pandas as pd

from volatility_trading.backtesting.reporting.plots import (
    plot_drawdown,
    plot_equity_vs_benchmark,
    plot_greeks_exposure,
    plot_margin_account,
    plot_performance_dashboard,
    plot_pnl_attribution,
    plot_rolling_metrics,
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
            "delta_pnl": [0.0, 1.0, -1.5],
            "Delta_PnL": [0.0, 0.5, -0.7],
            "Gamma_PnL": [0.0, 0.1, 0.05],
            "Vega_PnL": [0.0, 0.2, -0.3],
            "Theta_PnL": [0.0, -0.1, -0.1],
            "Other_PnL": [0.0, 0.3, -0.2],
        },
        index=index,
    )


def _sample_factor_mtm_daily() -> pd.DataFrame:
    frame = _sample_mtm_daily().copy()
    frame["IV_Level_PnL"] = [0.0, 0.15, -0.20]
    frame["RR_Skew_PnL"] = [0.0, 0.05, -0.10]
    frame["IV_Level_Prev_PnL"] = [0.0, 0.0, 0.0]
    frame["Vega_PnL"] = frame["IV_Level_PnL"] + frame["RR_Skew_PnL"]
    return frame


def _sample_benchmark(index: pd.Index) -> pd.Series:
    return pd.Series([3000.0, 3020.0, 3010.0], index=index)


def _sample_margin_diagnostics() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {
            "equity": [100_000.0, 99_000.0, 100_500.0],
            "initial_margin_requirement": [10_000.0, 10_000.0, 9_000.0],
            "maintenance_margin_requirement": [8_000.0, 8_000.0, 7_200.0],
            "margin_excess": [92_000.0, -500.0, 93_300.0],
            "margin_deficit": [0.0, 500.0, 0.0],
            "in_margin_call": [False, True, False],
            "forced_liquidation": [False, True, False],
        },
        index=index,
    )


def _sample_rolling_metrics() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "strategy_rolling_sharpe": [np.nan, np.nan, 0.5, 0.7, 0.9],
            "benchmark_rolling_sharpe": [np.nan, np.nan, 0.3, 0.4, 0.5],
            "strategy_rolling_annualized_volatility": [
                np.nan,
                np.nan,
                0.12,
                0.13,
                0.11,
            ],
            "benchmark_rolling_annualized_volatility": [
                np.nan,
                np.nan,
                0.08,
                0.09,
                0.10,
            ],
        },
        index=index,
    )


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


def test_plot_margin_account_returns_two_panel_figure():
    fig = plot_margin_account(_sample_margin_diagnostics())

    assert len(fig.axes) == 2


def test_plot_rolling_metrics_returns_two_panel_figure():
    fig = plot_rolling_metrics(_sample_rolling_metrics(), benchmark_name="SP500TR")

    assert len(fig.axes) == 2


def test_plot_pnl_attribution_returns_figure():
    mtm_daily = _sample_mtm_daily()
    fig_attr = plot_pnl_attribution(daily_mtm=mtm_daily)
    assert len(fig_attr.axes) == 1


def test_plot_pnl_attribution_accepts_attribution_only_frame():
    mtm_daily = _sample_mtm_daily().drop(columns=["equity"])

    fig_attr = plot_pnl_attribution(daily_mtm=mtm_daily)

    assert len(fig_attr.axes) == 1


def test_plot_pnl_attribution_prefers_factor_vol_components_when_available():
    mtm_daily = _sample_factor_mtm_daily()

    fig_attr = plot_pnl_attribution(daily_mtm=mtm_daily)

    labels = [line.get_label() for line in fig_attr.axes[0].lines]
    assert "IV_Level_PnL" in labels
    assert "RR_Skew_PnL" in labels
    assert "IV_Level_Prev_PnL" not in labels
    assert "Vega_PnL" not in labels


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
