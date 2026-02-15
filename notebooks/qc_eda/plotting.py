"""Plotting utilities for the ORATS SPY QC EDA notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from volatility_trading.iv_surface.term_structure import pick_closest_dte


def plot_smiles_by_delta(
    df_long: pl.DataFrame,
    picked_dates: Sequence[date],
    targets: Sequence[tuple[int, str]] = (
        (10, "10D"),
        (30, "30D"),
        (60, "60D"),
    ),
    *,
    nrows: int = 3,
    ncols: int = 3,
    event_labels: Mapping[date, str] | None = None,
) -> None:
    """Plot smoothed IV smiles vs |delta| for several DTE targets on multiple dates."""
    fig, axes = _make_facet_axes(picked_dates, nrows, ncols)

    for ax, day in zip(axes, picked_dates):
        sub = df_long.filter(pl.col("trade_date") == day)
        if sub.height == 0:
            ax.set_axis_off()
            continue

        dtes = sub.select(pl.col("dte").unique()).sort("dte").to_series().to_list()

        chosen: dict[str, int] = {}
        for target, label in targets:
            best = pick_closest_dte(dtes, target, max_tol=10)
            if best is not None:
                chosen[label] = best

        if not chosen:
            ax.set_axis_off()
            continue

        for label, dte_val in chosen.items():
            grp = (
                sub.filter(pl.col("dte") == dte_val)
                .with_columns(abs_delta=pl.col("put_delta").abs())
                .sort("abs_delta")
            )
            if grp.height == 0:
                continue

            ax.plot(
                grp["abs_delta"].to_numpy(),
                grp["smoothed_iv"].to_numpy(),
                label=f"{label} (DTE={dte_val})",
                linewidth=1.5,
            )

        ax.axvline(0.5, linestyle="--", linewidth=0.8)
        ax.set_xlabel("|Delta|")
        ax.set_ylabel("Smoothed IV")
        _set_panel_title(ax, day, event_labels)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for index in range(len(picked_dates), len(axes)):
        axes[index].set_axis_off()

    labels = ", ".join(label for _, label in targets)
    fig.suptitle(f"Smoothed IV Smiles by |Delta| ({labels})", y=1.01, fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_term_structures_by_delta(
    df: pl.DataFrame,
    picked_dates: Sequence[date],
    delta_targets: Sequence[tuple[float, str]] = (
        (0.50, "ATM"),
        (0.25, "25Δ Call"),
        (0.75, "25Δ Put"),
    ),
    *,
    nrows: int = 3,
    ncols: int = 3,
    event_labels: Mapping[date, str] | None = None,
) -> None:
    """Plot IV term structures for selected delta buckets across multiple dates."""
    fig, axes = _make_facet_axes(picked_dates, nrows, ncols)

    for ax, day in zip(axes, picked_dates):
        sub = df.filter(pl.col("trade_date") == day)
        if sub.height == 0:
            ax.set_axis_off()
            continue

        for target_delta, label in delta_targets:
            ts = (
                sub.with_columns(
                    (pl.col("call_delta") - target_delta).abs().alias("_dist")
                )
                .sort(["dte", "_dist"])
                .group_by("dte", maintain_order=True)
                .agg(pl.col("smoothed_iv").first().alias("smoothed_iv"))
                .sort("dte")
            )

            if ts.height == 0:
                continue

            dte_vals = ts["dte"].to_numpy()
            iv_vals = ts["smoothed_iv"].to_numpy()
            ax.plot(dte_vals, iv_vals, label=label, linewidth=1.5)
            ax.scatter(dte_vals, iv_vals, s=15, alpha=0.7, color="red")

        ax.set_xlabel("DTE")
        ax.set_ylabel("Smoothed IV")
        _set_panel_title(ax, day, event_labels)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for index in range(len(picked_dates), len(axes)):
        axes[index].set_axis_off()

    labels = ", ".join(label for _, label in delta_targets)
    fig.suptitle(f"IV Term Structures by Delta Bucket ({labels})", y=1.01, fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_avg_volume_by_delta(
    df_long: pl.DataFrame,
    delta_bins: np.ndarray | None = None,
    *,
    dte_min: int = 10,
    dte_max: int = 60,
    label_step: int = 2,
) -> None:
    """Plot average option volume by |delta| bucket, separately for calls and puts."""
    if delta_bins is None:
        delta_bins = np.linspace(0.0, 1.0, 21)
    delta_breaks: list[float] = np.asarray(delta_bins, dtype=float).tolist()

    vol_by_delta = (
        df_long.filter(pl.col("dte").is_between(dte_min, dte_max))
        .with_columns(delta_bucket=pl.col("delta").abs().cut(delta_breaks))
        .group_by(["delta_bucket", "option_type"])
        .agg(pl.col("volume").mean().alias("avg_volume"))
        .sort(["delta_bucket", "option_type"])
        .to_pandas()
    )

    if vol_by_delta.empty:
        return

    pivot = vol_by_delta.pivot_table(
        index="delta_bucket",
        columns="option_type",
        values="avg_volume",
    )

    edges = np.asarray(delta_bins)
    labels = [f"{edges[idx]:.2f}-{edges[idx + 1]:.2f}" for idx in range(len(edges) - 1)]
    x_values = np.arange(len(pivot))

    plt.figure(figsize=(8, 4))
    if "C" in pivot.columns:
        plt.plot(x_values, pivot["C"], marker="o", label="Call volume")
    if "P" in pivot.columns:
        plt.plot(x_values, pivot["P"], marker="o", label="Put volume")

    step = max(label_step, 1)
    plt.xticks(x_values[::step], labels[::step], rotation=25)
    plt.xlabel("|Delta| bucket")
    plt.ylabel("Average volume")
    plt.title("Average Option Volume by |Delta|")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_liquidity_by_dte(
    df_long: pl.DataFrame,
    *,
    dte_bins: Sequence[int] | None = None,
    dte_labels: Sequence[str] | None = None,
    delta_min: float = 0.1,
    delta_max: float = 0.9,
    delta_col: str = "delta",
) -> None:
    """Plot average volume and average open interest across DTE buckets."""
    if dte_bins is None:
        dte_bins = [0, 10, 15, 20, 30, 45, 60, 90, 180]
    dte_bins = list(dte_bins)

    if dte_labels is None:
        dte_labels = [
            f"({dte_bins[idx]}–{dte_bins[idx + 1]}]" for idx in range(len(dte_bins) - 1)
        ]
    else:
        dte_labels = list(dte_labels)

    if len(dte_labels) != len(dte_bins) - 1:
        raise ValueError("dte_labels must have length len(dte_bins)-1")

    bucket_expr = (
        pl.sum_horizontal([(pl.col("dte") > edge).cast(pl.Int32) for edge in dte_bins])
        - 1
    ).clip(0, len(dte_bins) - 2)

    liq_by_dte = (
        df_long.filter(
            pl.col(delta_col).abs().is_between(delta_min, delta_max),
            pl.col("dte").is_not_null(),
        )
        .with_columns(dte_bucket=bucket_expr)
        .group_by("dte_bucket")
        .agg(
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("open_interest").mean().alias("avg_open_interest"),
        )
    )

    all_buckets = pl.DataFrame({"dte_bucket": list(range(len(dte_labels)))})
    liq_by_dte = (
        all_buckets.join(liq_by_dte, on="dte_bucket", how="left")
        .with_columns(
            pl.col("avg_volume").fill_null(0.0),
            pl.col("avg_open_interest").fill_null(0.0),
        )
        .sort("dte_bucket")
    )

    x_values = np.arange(len(dte_labels))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.bar(
        x_values - 0.15,
        liq_by_dte["avg_volume"].to_numpy(),
        width=0.3,
        label="Avg volume",
    )
    ax2.bar(
        x_values + 0.15,
        liq_by_dte["avg_open_interest"].to_numpy(),
        width=0.3,
        alpha=0.7,
        label="Avg open interest",
    )

    ax1.set_xticks(x_values)
    ax1.set_xticklabels(dte_labels)
    ax1.set_xlabel("DTE bucket")
    ax1.set_ylabel("Avg volume")
    ax2.set_ylabel("Avg open interest")
    ax1.set_title("Liquidity vs time to expiry")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_spot_vs_yahoo(spx: pd.DataFrame) -> None:
    """Plot ORATS spot versus Yahoo close over time."""
    if spx.empty:
        return
    spx.plot(figsize=(12, 6))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("ORATS Spot vs Yahoo SPY Close")
    plt.tight_layout()
    plt.show()


def plot_term_structure_samples(
    df: pl.DataFrame,
    sample_days: Sequence[date],
    *,
    value_col: str = "dividend_yield",
    ex_div_dates: Sequence[date] | None = None,
    ex_div_label: str = "Ex-Div Date",
    nrows: int = 2,
    ncols: int = 2,
) -> None:
    """Plot a term structure (value vs DTE) for sample dates.

    For each selected `trade_date`, plots `value_col` against `dte`.

    Args:
        df: Polars DataFrame with at least ["trade_date", "dte", value_col].
        sample_days: Dates to plot (one panel per date).
        value_col: Column to plot on the y-axis (e.g., "dividend_yield",
            "risk_free_rate", or any numeric column).
        ex_div_dates: Optional ex-dividend calendar dates. When provided, the
            function overlays vertical dashed lines at matching future DTEs
            for each sample day.
        ex_div_label: Legend label for ex-dividend marker lines.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"trade_date", "dte", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    axes = np.asarray(axes).ravel()

    for ax, day in zip(axes, sample_days):
        sub = (
            df.filter(pl.col("trade_date") == day).select("dte", value_col).sort("dte")
        )
        if sub.height == 0:
            ax.set_axis_off()
            continue

        sub_pd = sub.to_pandas()
        ax.plot(sub_pd["dte"], sub_pd[value_col], marker="o", linestyle="-")

        if ex_div_dates:
            max_dte = int(sub_pd["dte"].max())
            future_ex_div_dtes = sorted(
                {
                    (ex_div_day - day).days
                    for ex_div_day in ex_div_dates
                    if 0 < (ex_div_day - day).days <= max_dte
                }
            )
            for idx, ex_div_dte in enumerate(future_ex_div_dtes):
                label = ex_div_label if idx == 0 else None
                ax.axvline(
                    ex_div_dte,
                    linestyle="--",
                    linewidth=1.0,
                    color="crimson",
                    alpha=0.35,
                    label=label,
                )

        ax.set_title(day.strftime("%Y-%m-%d"))
        ax.set_xlabel("DTE")
        ax.grid(alpha=0.3)
        if ex_div_dates:
            ax.legend(loc="lower right", frameon=False)

    for i in range(len(sample_days), len(axes)):
        axes[i].set_axis_off()

    if len(axes) > 0:
        axes[0].set_ylabel(value_col)

    fig.suptitle(f"Term-structure of {value_col} on sample days")
    fig.tight_layout()
    plt.show()


def plot_iv_time_series_with_slope(
    daily_features: pd.DataFrame,
    *,
    event_labels: Mapping[date, str] | None = None,
    columns: Sequence[str] = ("iv_10d", "iv_30d", "iv_90d", "iv_1y"),
) -> None:
    """Plot IV time series across maturities with a 10D-1Y slope panel."""
    iv_ts = daily_features.loc[:, list(columns)].dropna()
    if iv_ts.empty:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    series_style: dict[str, tuple[str, str, float]] = {
        "iv_10d": ("IV 10D", "#d62728", 1.4),
        "iv_30d": ("IV 30D", "#1f77b4", 1.4),
        "iv_90d": ("IV 90D", "#2ca02c", 1.4),
        "iv_1y": ("IV 1Y", "#9467bd", 1.6),
    }

    for column in columns:
        if column not in iv_ts.columns or column not in series_style:
            continue
        label, color, linewidth = series_style[column]
        ax_top.plot(
            iv_ts.index, iv_ts[column], label=label, color=color, linewidth=linewidth
        )

    if event_labels:
        for event_day in event_labels:
            ax_top.axvline(event_day, color="gray", alpha=0.15, linewidth=0.8)

    ax_top.set_title("Smoothed IV term snapshots over time (10D, 30D, 90D, 1Y)")
    ax_top.set_ylabel("Implied volatility")
    ax_top.grid(alpha=0.25)
    ax_top.legend(ncol=4, frameon=False, loc="upper left")

    iv_slope_10d_1y = iv_ts["iv_10d"] - iv_ts["iv_1y"]
    ax_bottom.plot(
        iv_ts.index,
        iv_slope_10d_1y,
        color="#111111",
        linewidth=1.2,
        label="10D - 1Y",
    )
    ax_bottom.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    if event_labels:
        for event_day in event_labels:
            ax_bottom.axvline(event_day, color="gray", alpha=0.15, linewidth=0.8)

    ax_bottom.set_ylabel("Slope")
    ax_bottom.set_xlabel("Trade date")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    plt.show()


def plot_greeks_vs_strike(
    df_long: pl.DataFrame,
    *,
    day: date,
    dte_target: int = 30,
    max_tol: int = 10,
) -> None:
    """Plot delta/gamma/vega/theta by strike for calls and puts."""
    sub = df_long.filter(pl.col("trade_date") == day)
    dtes_for_day = sub.select(pl.col("dte").unique()).sort("dte").to_series().to_list()

    dte_true = pick_closest_dte(dtes_for_day, dte_target, max_tol=max_tol)
    if dte_true is None:
        raise ValueError(
            f"No DTE within {max_tol} days of target={dte_target} on {day}"
        )

    sub = sub.filter(pl.col("dte") == dte_true).sort("strike")
    spot = float(sub.select("underlying_price").to_series().item(0))
    calls = sub.filter(pl.col("option_type") == "C")
    puts = sub.filter(pl.col("option_type") == "P")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_d, ax_g = axes[0]
    ax_v, ax_t = axes[1]

    plots: list[tuple[str, Axes, str, str]] = [
        ("delta", ax_d, "Delta", "Delta vs strike"),
        ("gamma", ax_g, "Gamma", "Gamma vs strike"),
        ("vega", ax_v, "Vega", "Vega vs strike"),
        ("theta", ax_t, "Theta", "Theta vs strike"),
    ]

    for col, ax, ylabel, title in plots:
        ax.plot(calls["strike"], calls[col], label=f"Call {col}", marker="o")
        ax.plot(puts["strike"], puts[col], label=f"Put {col}", marker="o")
        ax.axvline(spot, linestyle="--", linewidth=0.8, label="Spot S")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax_v.set_xlabel("Strike")
    ax_t.set_xlabel("Strike")
    fig.suptitle(f"Greeks vs strike - {day}, DTE={dte_true}, S={spot:.2f}")
    fig.tight_layout()
    plt.show()


def _make_facet_axes(
    picked_dates: Sequence[date],
    nrows: int,
    ncols: int,
    figsize_per: tuple[float, float] = (5.0, 4.0),
) -> tuple[Figure, np.ndarray]:
    """Create a faceted grid of subplots and flatten the axes array."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        sharex=False,
        sharey=False,
    )
    return fig, axes.ravel()


def _set_panel_title(
    ax: Axes,
    day: date,
    event_labels: Mapping[date, str] | None,
) -> None:
    """Set subplot title and append an event label when available."""
    title = day.strftime("%Y-%m-%d")
    if event_labels and day in event_labels:
        title += f" - {event_labels[day]}"
    ax.set_title(title)
