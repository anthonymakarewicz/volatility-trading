from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


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
    option_type: Literal["C", "P"] = "P",  # use puts to get equity-style skew
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
        dtes = (
            sub.select(pl.col("dte").unique())
            .sort("dte")
            .to_series()
            .to_list()
        )

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
                .with_columns(abs_delta = pl.col("put_delta").abs())
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
                    (pl.col("call_delta") - target_delta)
                    .abs()
                    .alias("_dist")
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


# ======================================================================
# Internal helpers
# ======================================================================

def pick_closest_dte(
    dtes: Sequence[int],
    target: int,
    max_tol: int = 5,
) -> int | None:
    """Return the DTE in `dtes` closest to `target` within `max_tol` days."""
    if not dtes:
        return None
    best = min(dtes, key=lambda d: abs(d - target))
    return best if abs(best - target) <= max_tol else None


def _make_facet_axes(
    picked_dates: Sequence[date],
    nrows: int,
    ncols: int,
    figsize_per: tuple[float, float] = (5.0, 4.0),
) -> tuple[plt.Figure, np.ndarray]:
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
    ax: plt.Axes,
    day: date,
    event_labels: Mapping[date, str] | None,
) -> None:
    """Set the subplot title, optionally appending an event label."""
    title = day.strftime("%Y-%m-%d")
    if event_labels and day in event_labels:
        title += f" – {event_labels[day]}"
    ax.set_title(title)