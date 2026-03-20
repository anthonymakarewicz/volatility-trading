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


def _resolve_pnl_attribution_columns(frame: pd.DataFrame) -> list[str]:
    """Return ordered P&L attribution component columns for plotting.

    When explicit factor-level vol columns are present, they replace the
    aggregate `Vega_PnL` line so the plot shows the more informative split.
    """
    columns = list(frame.columns)
    factor_pnl_columns = [
        column
        for column in columns
        if column.endswith("_PnL")
        and not column.endswith("_Prev_PnL")
        and column
        not in {
            "delta_pnl",
            "Delta_PnL",
            "Unhedged_Delta_PnL",
            "Gamma_PnL",
            "Vega_PnL",
            "Theta_PnL",
            "Other_PnL",
        }
    ]

    ordered = ["Delta_PnL", "Gamma_PnL"]
    if factor_pnl_columns:
        ordered.extend(sorted(factor_pnl_columns))
    else:
        ordered.append("Vega_PnL")
    ordered.extend(["Theta_PnL", "Other_PnL"])
    return ordered


def _rebased_benchmark(
    benchmark: pd.Series | None, equity: pd.Series
) -> pd.Series | None:
    if benchmark is None:
        return None
    bm = pd.Series(benchmark).copy()
    bm.index = pd.to_datetime(bm.index)
    bm = bm.sort_index().reindex(equity.index).ffill()
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


def plot_margin_account(margin_diagnostics: pd.DataFrame) -> Figure:
    """Return a two-panel figure for margin account state and call events."""
    if margin_diagnostics.empty or "equity" not in margin_diagnostics.columns:
        raise ValueError(
            "margin_diagnostics must be non-empty and contain an 'equity' column"
        )

    frame = margin_diagnostics.copy()
    frame.index = pd.to_datetime(frame.index)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_top, ax_bottom = axes

    ax_top.plot(frame.index, frame["equity"].astype(float), label="Equity")
    ax_top.plot(
        frame.index,
        frame["initial_margin_requirement"].astype(float),
        label="Initial Margin",
    )
    ax_top.plot(
        frame.index,
        frame["maintenance_margin_requirement"].astype(float),
        label="Maintenance Margin",
    )
    ax_top.set_title("Margin Account")
    ax_top.set_ylabel("USD")
    ax_top.legend(loc="upper left")
    ax_top.grid(True)

    margin_excess = frame["margin_excess"].astype(float)
    margin_deficit = frame["margin_deficit"].astype(float)
    ax_bottom.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_bottom.plot(frame.index, margin_excess, label="Margin Excess", color="tab:green")
    ax_bottom.plot(frame.index, margin_deficit, label="Margin Deficit", color="tab:red")

    margin_call_mask = frame["in_margin_call"].astype(bool)
    forced_liquidation_mask = frame["forced_liquidation"].astype(bool)
    if margin_call_mask.any():
        ax_bottom.scatter(
            frame.index[margin_call_mask],
            margin_excess[margin_call_mask],
            label="Margin Call",
            color="tab:orange",
            marker="x",
            zorder=3,
        )
    if forced_liquidation_mask.any():
        ax_bottom.scatter(
            frame.index[forced_liquidation_mask],
            margin_excess[forced_liquidation_mask],
            label="Forced Liquidation",
            color="tab:red",
            marker="o",
            zorder=4,
        )

    ax_bottom.set_ylabel("USD")
    ax_bottom.set_xlabel("Date")
    ax_bottom.legend(loc="lower left")
    ax_bottom.grid(True)
    fig.tight_layout()
    return fig


def plot_rolling_metrics(
    rolling_metrics: pd.DataFrame,
    *,
    benchmark_name: str = "Benchmark",
) -> Figure:
    """Return rolling Sharpe and annualized volatility panels."""
    if rolling_metrics.empty:
        raise ValueError("rolling_metrics must be non-empty")

    frame = rolling_metrics.copy()
    frame.index = pd.to_datetime(frame.index)
    if "strategy_rolling_sharpe" not in frame.columns:
        raise ValueError("rolling_metrics must contain strategy rolling metric columns")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_top, ax_bottom = axes

    ax_top.plot(
        frame.index,
        frame["strategy_rolling_sharpe"].astype(float),
        label="Strategy Sharpe",
        color="tab:blue",
    )
    if "benchmark_rolling_sharpe" in frame.columns:
        ax_top.plot(
            frame.index,
            frame["benchmark_rolling_sharpe"].astype(float),
            label=f"{benchmark_name} Sharpe",
            color="tab:orange",
        )
    ax_top.set_title("Rolling Metrics")
    ax_top.set_ylabel("Rolling Sharpe")
    ax_top.legend(loc="upper left")
    ax_top.grid(True)

    ax_bottom.plot(
        frame.index,
        frame["strategy_rolling_annualized_volatility"].astype(float),
        label="Strategy Volatility",
        color="tab:blue",
    )
    if "benchmark_rolling_annualized_volatility" in frame.columns:
        ax_bottom.plot(
            frame.index,
            frame["benchmark_rolling_annualized_volatility"].astype(float),
            label=f"{benchmark_name} Volatility",
            color="tab:orange",
        )
    ax_bottom.set_ylabel("Rolling Ann. Vol")
    ax_bottom.set_xlabel("Date")
    ax_bottom.legend(loc="upper left")
    ax_bottom.grid(True)

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
    figsize: tuple | None = None,
) -> Figure:
    """Build a dashboard figure with equity, drawdown, and Greek exposures."""
    equity = _require_equity(mtm_daily)
    bm_rebase = _rebased_benchmark(benchmark, equity)

    figsize = figsize if figsize is not None else (14, 14)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
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


def plot_pnl_attribution(
    daily_mtm: pd.DataFrame, figsize: tuple | None = None
) -> Figure:
    """Return cumulative total and Greek-attribution PnL figure."""
    if daily_mtm.empty:
        raise ValueError("daily_mtm must be non-empty")

    frame = daily_mtm.copy()
    frame.index = pd.to_datetime(frame.index)
    cumulative = pd.DataFrame(index=frame.index)
    if "equity" in frame.columns:
        equity = _require_equity(frame)
        cumulative["Total P&L"] = equity - float(equity.iloc[0])
    elif "delta_pnl" in frame.columns:
        cumulative["Total P&L"] = frame["delta_pnl"].astype(float).cumsum()
    else:
        raise ValueError("daily_mtm must contain either 'equity' or 'delta_pnl'")

    pnl_columns = _resolve_pnl_attribution_columns(frame)
    for column in pnl_columns:
        if column in frame.columns:
            cumulative[column] = frame[column].astype(float).cumsum()
        else:
            cumulative[column] = 0.0

    figsize = figsize if figsize is not None else (12, 5)
    base_colors = {
        "Total P&L": "purple",
        "Delta_PnL": "red",
        "Gamma_PnL": "orange",
        "Vega_PnL": "green",
        "Theta_PnL": "blue",
        "Other_PnL": "brown",
        "IV_Level_PnL": "green",
        "RR_Skew_PnL": "teal",
    }
    fallback_cycle = [
        "tab:green",
        "tab:cyan",
        "tab:pink",
        "tab:gray",
        "tab:olive",
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        cumulative.index,
        cumulative["Total P&L"],
        label="Total P&L",
        color=base_colors["Total P&L"],
    )
    for idx, column in enumerate(pnl_columns):
        ax.plot(
            cumulative.index,
            cumulative[column],
            label=column,
            color=base_colors.get(column, fallback_cycle[idx % len(fallback_cycle)]),
        )

    ax.set_title("Cumulative P&L Attribution: Total vs Component Contributions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.legend()
    fig.tight_layout()
    return fig


# TODO(reporting): add optional hedge decomposition panel
# (cum_hedge_carry, cum_hedge_cost, cum_net_hedge, cum_hedging_effect)
# in follow-up branch: feature/hedge-attribution-reporting.


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
    figsize: tuple | None = None,
) -> Figure:
    """Return actual and scenario-stressed equity curves."""
    equity = _require_equity(daily_mtm)
    stress_columns = _resolve_stress_columns(stressed_mtm, scenarios)
    if not stress_columns:
        raise ValueError("No matching stress scenario columns found in stressed_mtm")

    figsize = figsize if figsize is not None else (12, 5)
    fig, ax = plt.subplots(figsize=figsize)
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
