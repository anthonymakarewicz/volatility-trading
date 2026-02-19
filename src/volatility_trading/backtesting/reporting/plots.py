"""Plot builders for backtest reporting artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _require_equity(mtm_daily: pd.DataFrame) -> pd.Series:
    if mtm_daily.empty or "equity" not in mtm_daily.columns:
        raise ValueError("mtm_daily must be non-empty and contain an 'equity' column")
    return mtm_daily["equity"].astype(float)


def _rebased_benchmark(
    benchmark: pd.Series | None, equity: pd.Series
) -> pd.Series | None:
    if benchmark is None:
        return None
    bm = benchmark.reindex(equity.index).ffill()
    if bm.isna().all() or float(bm.iloc[0]) == 0.0:
        return None
    return (bm / bm.iloc[0]) * equity.iloc[0]


def _plot_equity_panel(
    *,
    ax: plt.Axes,
    equity: pd.Series,
    benchmark_rebased: pd.Series | None,
    strategy_name: str,
    benchmark_name: str,
) -> None:
    ax.plot(equity.index, equity, color="tab:blue", label=f"{strategy_name} equity")
    if benchmark_rebased is not None:
        ax.plot(
            benchmark_rebased.index,
            benchmark_rebased,
            color="tab:orange",
            label=f"{benchmark_name} (rebased)",
        )
        ax.set_title(f"Equity vs. {benchmark_name}")
    else:
        ax.set_title("Equity")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left")
    ax.grid(True)


def _plot_drawdown_panel(
    *,
    ax: plt.Axes,
    equity: pd.Series,
    benchmark_rebased: pd.Series | None,
    benchmark_name: str,
) -> None:
    strat_dd = (equity - equity.cummax()) / equity.cummax()
    ax.fill_between(
        strat_dd.index, strat_dd, 0, color="tab:blue", alpha=0.3, label="Strategy DD"
    )
    if benchmark_rebased is not None:
        bm_dd = (
            benchmark_rebased - benchmark_rebased.cummax()
        ) / benchmark_rebased.cummax()
        ax.fill_between(
            bm_dd.index,
            bm_dd,
            0,
            color="tab:orange",
            alpha=0.3,
            label=f"{benchmark_name} DD",
        )
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True)


def _plot_greeks_panels(
    *,
    axes: list[plt.Axes],
    mtm_daily: pd.DataFrame,
    show_xlabel_on_last_row: bool,
) -> None:
    greek_cols = ["net_delta", "gamma", "vega", "theta"]
    greek_titles = [
        "Total Delta Exposure",
        "Total Gamma Exposure",
        "Total Vega Exposure",
        "Total Theta Exposure",
    ]
    greek_colors = ["red", "orange", "green", "blue"]
    for i, (column, greek_title, color) in enumerate(
        zip(greek_cols, greek_titles, greek_colors)
    ):
        ax = axes[i]
        series = (
            mtm_daily[column].astype(float)
            if column in mtm_daily.columns
            else pd.Series(np.zeros(len(mtm_daily), dtype=float), index=mtm_daily.index)
        )
        ax.plot(mtm_daily.index, series, color=color)
        ax.set_title(greek_title)
        ax.set_ylabel(column)
        if show_xlabel_on_last_row and i >= 2:
            ax.set_xlabel("Date")
        ax.grid(True)


def plot_equity_vs_benchmark(
    benchmark: pd.Series | None,
    mtm_daily: pd.DataFrame,
    *,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> Figure:
    """Return a standalone equity-vs-benchmark figure."""
    equity = _require_equity(mtm_daily)
    benchmark_rebased = _rebased_benchmark(benchmark, equity)

    fig, ax = plt.subplots(figsize=(12, 5))
    _plot_equity_panel(
        ax=ax,
        equity=equity,
        benchmark_rebased=benchmark_rebased,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
    )
    fig.tight_layout()
    return fig


def plot_drawdown(
    benchmark: pd.Series | None,
    mtm_daily: pd.DataFrame,
    *,
    benchmark_name: str = "Benchmark",
) -> Figure:
    """Return a standalone drawdown figure."""
    equity = _require_equity(mtm_daily)
    benchmark_rebased = _rebased_benchmark(benchmark, equity)

    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_drawdown_panel(
        ax=ax,
        equity=equity,
        benchmark_rebased=benchmark_rebased,
        benchmark_name=benchmark_name,
    )
    fig.tight_layout()
    return fig


def plot_greeks_exposure(mtm_daily: pd.DataFrame) -> Figure:
    """Return a standalone 2x2 Greeks exposure figure."""
    _require_equity(mtm_daily)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    _plot_greeks_panels(
        axes=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        mtm_daily=mtm_daily,
        show_xlabel_on_last_row=True,
    )
    fig.tight_layout()
    return fig


def plot_performance_dashboard(
    benchmark: pd.Series | None,
    mtm_daily: pd.DataFrame,
    *,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    title: str | None = None,
    subtitle: str | None = None,
) -> Figure:
    """Build a dashboard figure with equity, drawdown, and Greek exposures."""
    equity = _require_equity(mtm_daily)
    bm_rebase = _rebased_benchmark(benchmark, equity)

    fig = plt.figure(figsize=(14, 14), constrained_layout=False)
    grid = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.3)
    fig.subplots_adjust(top=0.90)

    fig.suptitle(
        title or f"{strategy_name} - Performance Dashboard",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.955,
        subtitle or "Equity vs benchmark, drawdowns, and portfolio Greeks over time",
        ha="center",
        va="top",
        fontsize=11,
        color="dimgray",
    )

    ax0 = fig.add_subplot(grid[0, :])
    _plot_equity_panel(
        ax=ax0,
        equity=equity,
        benchmark_rebased=bm_rebase,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
    )

    ax1 = fig.add_subplot(grid[1, :])
    _plot_drawdown_panel(
        ax=ax1,
        equity=equity,
        benchmark_rebased=bm_rebase,
        benchmark_name=benchmark_name,
    )

    greek_axes = [
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
    ]
    _plot_greeks_panels(
        axes=greek_axes,
        mtm_daily=mtm_daily,
        show_xlabel_on_last_row=True,
    )

    return fig


def plot_pnl_attribution(daily_mtm: pd.DataFrame) -> Figure:
    """Return cumulative total and Greek-attribution PnL figure."""
    equity = _require_equity(daily_mtm)
    cumulative = pd.DataFrame(index=daily_mtm.index)
    cumulative["Total P&L"] = equity - float(equity.iloc[0])

    greek_columns = ["Delta_PnL", "Gamma_PnL", "Vega_PnL", "Theta_PnL", "Other_PnL"]
    for column in greek_columns:
        if column in daily_mtm.columns:
            cumulative[column] = daily_mtm[column].astype(float).cumsum()
        else:
            cumulative[column] = 0.0

    colors = {
        "Total P&L": "purple",
        "Delta_PnL": "red",
        "Gamma_PnL": "orange",
        "Vega_PnL": "green",
        "Theta_PnL": "blue",
        "Other_PnL": "brown",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        cumulative.index,
        cumulative["Total P&L"],
        label="Total P&L",
        color=colors["Total P&L"],
    )
    for column in greek_columns:
        ax.plot(
            cumulative.index,
            cumulative[column],
            label=column,
            color=colors[column],
        )

    ax.set_title("Cumulative P&L Attribution: Total vs Greek Contributions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


def _resolve_stress_columns(
    stressed_mtm: pd.DataFrame,
    scenarios: Mapping[str, object] | Sequence[str],
) -> list[str]:
    scenario_names = (
        list(scenarios.keys()) if isinstance(scenarios, Mapping) else list(scenarios)
    )
    columns: list[str] = []
    for name in scenario_names:
        if name in stressed_mtm.columns:
            columns.append(name)
            continue
        prefixed = f"PnL_{name}"
        if prefixed in stressed_mtm.columns:
            columns.append(prefixed)
    return columns


def plot_stressed_pnl(
    stressed_mtm: pd.DataFrame,
    daily_mtm: pd.DataFrame,
    scenarios: Mapping[str, object] | Sequence[str],
) -> Figure:
    """Return actual and scenario-stressed equity curves."""
    equity = _require_equity(daily_mtm)
    stress_columns = _resolve_stress_columns(stressed_mtm, scenarios)
    if not stress_columns:
        raise ValueError("No matching stress scenario columns found in stressed_mtm")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity, label="Actual Equity")
    for column in stress_columns:
        shocked = stressed_mtm[column].reindex(equity.index).fillna(0.0).cumsum()
        ax.plot(equity.index, equity + shocked, label=column)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.set_title("Equity Curve vs. Stressed Equity Curves")
    ax.legend()
    fig.tight_layout()
    return fig
