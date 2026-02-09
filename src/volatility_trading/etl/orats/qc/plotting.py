from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from volatility_trading.iv_surface.term_structure import pick_closest_dte

# ======================================================================
# Public API
# ======================================================================


def plot_smiles_by_delta(
    df_long: pl.DataFrame,
    picked_dates: Sequence[date],
    targets: Sequence[tuple[int, str]] = (
        (10, "≈10D"),
        (30, "≈30D"),
        (60, "≈60D"),
    ),
    *,
    nrows: int = 3,
    ncols: int = 3,
    event_labels: Mapping[date, str] | None = None,
) -> None:
    """
    Plot smoothed IV smiles vs |delta| for several DTE targets on multiple dates.

    Parameters
    ----------
    df :
        WIDE ORATS panel with columns at least:
        ["trade_date", "dte", "option_type", "call_delta", "smoothed_iv"].
    picked_dates :
        Dates to show as facets.
    targets :
        Sequence of (target_dte, label) pairs, e.g. (10, "≈10D").
    nrows, ncols :
        Grid shape for the facet layout.
    event_labels :
        Optional mapping {date -> label} to annotate special events in titles.
    option_type :
        "C" for calls or "P" for puts. For SPX-like equity index skew,
        using puts ("P") gives the usual downward-sloping smile in |delta|.
    """
    fig, axes = _make_facet_axes(picked_dates, nrows, ncols)

    for ax, day in zip(axes, picked_dates):
        sub = df_long.filter(pl.col("trade_date") == day)
        if sub.height == 0:
            ax.set_axis_off()
            continue

        # available DTEs that day
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
                sub.filter(
                    pl.col("dte") == dte_val,
                )
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

        ax.axvline(0.5, linestyle="--", linewidth=0.8)  # ATM call delta
        ax.set_xlabel("|Delta|")
        ax.set_ylabel("Smoothed IV")
        _set_panel_title(ax, day, event_labels)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    # hide unused axes (if fewer dates than grid slots)
    for k in range(len(picked_dates), len(axes)):
        axes[k].set_axis_off()

    fig.tight_layout()
    plt.show()


def plot_term_structures_by_delta(
    df: pl.DataFrame,
    picked_dates: Sequence[date],
    delta_targets: Sequence[tuple[float, str]] = (
        (0.50, "ATM (Δ≈0.5)"),
        (0.25, "25Δ Call"),
        (0.75, "25Δ Put"),
    ),
    *,
    nrows: int = 3,
    ncols: int = 3,
    event_labels: Mapping[date, str] | None = None,
) -> None:
    """
    Plot IV term structures for a few delta buckets across multiple dates.

    For each (trade_date, target_delta) and each DTE, we pick the option
    whose call_delta is closest to `target_delta` and plot smoothed_iv vs DTE.

    Parameters
    ----------
    df :
        WIDE ORATS panel with columns at least:
        ["trade_date", "dte", "call_delta", "smoothed_iv"].
    picked_dates :
        Dates to show as facets.
    delta_targets :
        Sequence of (delta, label) pairs to plot, e.g. (0.5, "ATM").
    nrows, ncols :
        Grid shape for the facet layout.
    event_labels :
        Optional mapping {date -> label} to annotate special events in titles.
    """
    fig, axes = _make_facet_axes(picked_dates, nrows, ncols)

    for ax, day in zip(axes, picked_dates):
        sub = df.filter(pl.col("trade_date") == day)
        if sub.height == 0:
            ax.set_axis_off()
            continue

        for target_delta, label in delta_targets:
            # for each DTE, pick row whose call_delta is closest to target_delta
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

    for k in range(len(picked_dates), len(axes)):
        axes[k].set_axis_off()

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
    """
    Plot average option volume by |delta| bucket, separately for calls and puts.

    Parameters
    ----------
    df_long :
        LONG ORATS panel with at least:
        ["dte", "delta", "option_type", "volume"].
    delta_bins :
        Array of bin edges for |delta|, e.g. np.linspace(0.0, 1.0, 21).
        If None, use 0.00–1.00 in steps of 0.05.
    dte_min, dte_max :
        Restrict to options with dte in [dte_min, dte_max].
    label_step :
        Show every `label_step`th bucket label on the x-axis for readability.
    """
    if delta_bins is None:
        delta_bins = np.linspace(0.0, 1.0, 21)
    delta_breaks: list[float] = np.asarray(delta_bins, dtype=float).tolist()

    vol_by_delta = (
        df_long.filter(pl.col("dte").is_between(dte_min, dte_max))
        .with_columns(
            delta_bucket=pl.col("delta").abs().cut(delta_breaks),
        )
        .group_by(["delta_bucket", "option_type"])
        .agg(pl.col("volume").mean().alias("avg_volume"))
        .sort(["delta_bucket", "option_type"])
        .to_pandas()
    )

    if vol_by_delta.empty:
        return

    pivot = vol_by_delta.pivot_table(
        index="delta_bucket",
        columns="option_type",  # "C" / "P"
        values="avg_volume",
    )

    # Build human-readable labels from bin edges
    edges = np.asarray(delta_bins)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]

    x = np.arange(len(pivot))

    plt.figure(figsize=(8, 4))

    if "C" in pivot.columns:
        plt.plot(x, pivot["C"], marker="o", label="Call volume")
    if "P" in pivot.columns:
        plt.plot(x, pivot["P"], marker="o", label="Put volume")

    # Avoid index/label mismatch if some bins are empty
    step = max(label_step, 1)
    plt.xticks(x[::step], labels[::step], rotation=25)

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
    delta_col: str = "delta",  # <-- change if your column is call_delta etc.
) -> None:
    if dte_bins is None:
        dte_bins = [0, 10, 15, 20, 30, 45, 60, 90, 180]
    dte_bins = list(dte_bins)

    if dte_labels is None:
        dte_labels = [
            f"({dte_bins[i]}–{dte_bins[i + 1]}]" for i in range(len(dte_bins) - 1)
        ]
    else:
        dte_labels = list(dte_labels)

    if len(dte_labels) != len(dte_bins) - 1:
        raise ValueError("dte_labels must have length len(dte_bins)-1")

    # --- 1) Build bucket index (0..n-1) without relying on cut() string labels ---
    # bucket = sum(dte > edge) - 1 (clipped)
    # Example: dte=12 with edges [0,10,15,...] => (dte>0)+(dte>10)=2 => bucket=1 => (10,15]
    bucket_expr = (
        pl.sum_horizontal([(pl.col("dte") > e).cast(pl.Int32) for e in dte_bins]) - 1
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

    # Ensure we always plot all buckets in order
    all_buckets = pl.DataFrame({"dte_bucket": list(range(len(dte_labels)))})
    liq_by_dte = (
        all_buckets.join(liq_by_dte, on="dte_bucket", how="left")
        .with_columns(
            pl.col("avg_volume").fill_null(0.0),
            pl.col("avg_open_interest").fill_null(0.0),
        )
        .sort("dte_bucket")
    )

    x = np.arange(len(dte_labels))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.bar(
        x - 0.15, liq_by_dte["avg_volume"].to_numpy(), width=0.3, label="Avg volume"
    )
    ax2.bar(
        x + 0.15,
        liq_by_dte["avg_open_interest"].to_numpy(),
        width=0.3,
        alpha=0.7,
        label="Avg open interest",
    )

    ax1.set_xticks(x)
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


# ======================================================================
# Internal helpers
# ======================================================================


def _make_facet_axes(
    picked_dates: Sequence[date],
    nrows: int,
    ncols: int,
    figsize_per: tuple[float, float] = (5.0, 4.0),
) -> tuple[Figure, np.ndarray]:
    """Helper to create a faceted grid of subplots and flatten the axes array."""
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
    """Set the subplot title, optionally appending an event label."""
    title = day.strftime("%Y-%m-%d")
    if event_labels and day in event_labels:
        title += f" – {event_labels[day]}"
    ax.set_title(title)
