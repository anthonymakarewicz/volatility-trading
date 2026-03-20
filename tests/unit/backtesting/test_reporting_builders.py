from typing import cast

import numpy as np
import pandas as pd
import pytest

from volatility_trading.backtesting.reporting.builders import (
    build_benchmark_comparison_payload,
    build_equity_and_drawdown_table,
    build_exposures_daily_table,
    build_margin_diagnostics_table,
    build_pnl_attribution_daily_table,
    build_rolling_metrics_table,
    build_summary_metrics,
    build_trades_table,
)

# TODO: Create subfolder for reporting/


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
            "factor_iv_level": [24.0, 24.5, 23.5],
            "factor_exposure_iv_level": [10.0, 11.0, 12.0],
            "factor_exposure_iv_level_prev": [0.0, 10.0, 11.0],
            "factor_rr_skew": [-11.0, -10.0, -9.0],
            "factor_exposure_rr_skew": [4.0, 4.5, 5.0],
            "factor_exposure_rr_skew_prev": [0.0, 4.0, 4.5],
            "delta_pnl": [0.0, 10.0, -11.0],
            "Delta_PnL": [0.0, 3.0, -4.0],
            "Unhedged_Delta_PnL": [0.0, 4.0, -5.0],
            "Gamma_PnL": [0.0, 1.0, 0.5],
            "Vega_PnL": [0.0, 2.0, -1.0],
            "Theta_PnL": [0.0, -0.5, -0.5],
            "Other_PnL": [0.0, 4.5, -6.0],
            "IV_Level_PnL": [0.0, 1.5, -2.0],
            "RR_Skew_PnL": [0.0, 0.5, 1.0],
        },
        index=index,
    )


def _sample_margin_mtm_daily() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {
            "equity": [100_000.0, 100_050.0, 100_030.0],
            "open_contracts": [2.0, np.nan, 2.0],
            "margin_per_contract": [500.0, np.nan, 500.0],
            "initial_margin_requirement": [1_000.0, np.nan, 1_000.0],
            "maintenance_margin_requirement": [800.0, np.nan, 800.0],
            "margin_excess": [99_200.0, np.nan, 99_230.0],
            "margin_deficit": [0.0, np.nan, 0.0],
            "in_margin_call": [False, np.nan, False],
            "margin_call_days": [0.0, np.nan, 0.0],
            "forced_liquidation": [False, np.nan, False],
            "contracts_liquidated": [0.0, np.nan, 0.0],
            "financing_pnl": [-1.25, np.nan, -1.1],
            "hedge_turnover": [10.0, np.nan, 5.0],
            "hedge_trade_count": [1.0, np.nan, 1.0],
        },
        index=index,
    )


def _sample_long_mtm_daily() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=70, freq="D")
    equity = pd.Series(100_000.0 + np.linspace(0.0, 3_000.0, len(index)), index=index)
    delta_pnl = equity.diff().fillna(0.0)
    return pd.DataFrame(
        {
            "equity": equity,
            "delta": np.linspace(0.0, 5.0, len(index)),
            "net_delta": np.linspace(0.5, 5.5, len(index)),
            "gamma": np.linspace(0.1, 0.3, len(index)),
            "vega": np.linspace(10.0, 15.0, len(index)),
            "theta": np.linspace(-1.0, -1.5, len(index)),
            "hedge_pnl": np.linspace(0.0, 2.0, len(index)),
            "delta_pnl": delta_pnl,
            "Delta_PnL": delta_pnl * 0.4,
            "Unhedged_Delta_PnL": delta_pnl * 0.5,
            "Gamma_PnL": delta_pnl * 0.1,
            "Vega_PnL": delta_pnl * 0.2,
            "Theta_PnL": delta_pnl * 0.05,
            "Other_PnL": delta_pnl * 0.25,
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
        "factor_exposure_iv_level",
        "factor_exposure_rr_skew",
    ]
    assert len(out) == len(mtm_daily)


def test_build_margin_diagnostics_table_forward_fills_state_and_zero_fills_events():
    mtm_daily = _sample_margin_mtm_daily()

    out = build_margin_diagnostics_table(mtm_daily)

    assert list(out.columns) == [
        "equity",
        "open_contracts",
        "margin_per_contract",
        "initial_margin_requirement",
        "maintenance_margin_requirement",
        "margin_excess",
        "margin_deficit",
        "initial_margin_utilization",
        "maintenance_margin_utilization",
        "in_margin_call",
        "margin_call_days",
        "forced_liquidation",
        "contracts_liquidated",
        "financing_pnl",
        "hedge_turnover",
        "hedge_trade_count",
    ]
    assert out.loc[pd.Timestamp("2020-01-02"), "open_contracts"] == 2
    assert out.loc[
        pd.Timestamp("2020-01-02"), "initial_margin_requirement"
    ] == pytest.approx(1_000.0)
    assert not bool(out.loc[pd.Timestamp("2020-01-02"), "forced_liquidation"])
    assert out.loc[pd.Timestamp("2020-01-02"), "contracts_liquidated"] == 0
    assert out.loc[pd.Timestamp("2020-01-02"), "financing_pnl"] == pytest.approx(0.0)
    assert out.loc[pd.Timestamp("2020-01-02"), "hedge_turnover"] == pytest.approx(0.0)
    assert out.loc[
        pd.Timestamp("2020-01-01"), "initial_margin_utilization"
    ] == pytest.approx(0.01)
    assert out.loc[
        pd.Timestamp("2020-01-01"), "maintenance_margin_utilization"
    ] == pytest.approx(0.008)


def test_build_rolling_metrics_table_respects_benchmark_presence():
    mtm_daily = _sample_long_mtm_daily()
    benchmark = pd.Series(
        np.linspace(3_000.0, 3_300.0, len(mtm_daily)),
        index=mtm_daily.index,
    )

    with_benchmark = build_rolling_metrics_table(
        mtm_daily=mtm_daily,
        benchmark=benchmark,
    )
    without_benchmark = build_rolling_metrics_table(
        mtm_daily=mtm_daily,
        benchmark=None,
    )

    assert "strategy_rolling_return" in with_benchmark.columns
    assert "strategy_rolling_annualized_volatility" in with_benchmark.columns
    assert "strategy_rolling_sharpe" in with_benchmark.columns
    assert "strategy_rolling_drawdown" in with_benchmark.columns
    assert "benchmark_rolling_return" in with_benchmark.columns
    assert "benchmark_rolling_annualized_volatility" in with_benchmark.columns
    assert "benchmark_rolling_sharpe" in with_benchmark.columns
    assert "benchmark_rolling_drawdown" in with_benchmark.columns
    assert "relative_equity_spread" in with_benchmark.columns
    assert "benchmark_rolling_return" not in without_benchmark.columns
    assert pd.notna(with_benchmark.iloc[-1]["strategy_rolling_return"])


def test_build_benchmark_comparison_payload_matches_expected_sections():
    mtm_daily = _sample_long_mtm_daily()
    benchmark = pd.Series(
        np.linspace(3_000.0, 3_300.0, len(mtm_daily)),
        index=mtm_daily.index,
    )

    out = build_benchmark_comparison_payload(
        mtm_daily=mtm_daily,
        benchmark=benchmark,
    )

    assert out is not None
    assert set(out) == {"strategy", "benchmark", "relative"}
    assert set(out["strategy"]) == {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    }
    assert set(out["relative"]) == {
        "total_return_diff",
        "cagr_diff",
        "annualized_volatility_diff",
        "sharpe_diff",
        "max_drawdown_diff",
    }


def test_build_benchmark_comparison_payload_omits_output_without_benchmark():
    mtm_daily = _sample_long_mtm_daily()

    out = build_benchmark_comparison_payload(mtm_daily=mtm_daily, benchmark=None)

    assert out is None


def test_build_pnl_attribution_daily_table_selects_expected_columns():
    mtm_daily = _sample_mtm_daily()

    out = build_pnl_attribution_daily_table(mtm_daily)

    assert list(out.columns) == [
        "delta_pnl",
        "Delta_PnL",
        "Unhedged_Delta_PnL",
        "Gamma_PnL",
        "Vega_PnL",
        "Theta_PnL",
        "Other_PnL",
        "IV_Level_PnL",
        "RR_Skew_PnL",
    ]
    assert out.loc[pd.Timestamp("2020-01-02"), "Delta_PnL"] == pytest.approx(3.0)


def test_build_trades_table_normalizes_trade_legs_payload():
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
            "exit_date": [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-04")],
            "contracts": [1, 2],
            "pnl": [1.0, -0.5],
            "trade_legs": [
                (
                    {
                        "leg_index": np.int64(0),
                        "expiry_date": pd.Timestamp("2020-01-31"),
                        "side": np.int64(-1),
                        "entry_price": np.float64(5.2),
                    },
                ),
                None,
            ],
        }
    )

    out = build_trades_table(trades)

    trade_legs_0 = cast(list[dict[str, object]], out.at[0, "trade_legs"])
    assert isinstance(trade_legs_0, list)
    assert trade_legs_0[0]["leg_index"] == 0
    assert trade_legs_0[0]["expiry_date"] == "2020-01-31T00:00:00"
    trade_legs_1 = cast(list[dict[str, object]], out.at[1, "trade_legs"])
    assert trade_legs_1 == []
