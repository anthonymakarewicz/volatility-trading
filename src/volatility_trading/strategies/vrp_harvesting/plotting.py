import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_vrp(iv_atm, rv, vrp):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot IV and RV
    axes[0].plot(iv_atm.index, iv_atm, label='30D ATM IV', color='tab:blue')
    axes[0].plot(rv.index, rv, label='21D Realized Vol', color='tab:orange')
    axes[0].set_ylabel('Volatility (%)')
    axes[0].legend()
    axes[0].set_title('30D ATM Implied Volatility vs. 21D Realized Volatility')

    # Plot VRP
    axes[1].plot(vrp.index, vrp, label='Volatility Risk Premium (IV - RV)', color='tab:green')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_ylabel('VRP (%)')
    axes[1].set_title('Volatility Risk Premium')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_vrp_hist(vrp):
    plt.figure(figsize=(8, 6))

    # Histogram with mean
    plt.hist(vrp.dropna(), bins=50, color='tab:green', alpha=0.7, edgecolor='black', density=True)
    plt.axvline(vrp.mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {vrp.mean():.2f}")
    plt.axvline(vrp.median(), color='blue', linestyle='--', linewidth=1.5, label=f"Median: {vrp.median():.2f}")

    # Add quantiles
    p5, p95 = np.percentile(vrp.dropna(), [5, 95])
    plt.axvline(p5, color='gray', linestyle='--', linewidth=1, label=f"5th %: {p5:.2f}")
    plt.axvline(p95, color='gray', linestyle='--', linewidth=1, label=f"95th %: {p95:.2f}")
    plt.xlabel('VRP (%)')
    plt.ylabel('Density')
    plt.title('VRP Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_vrp_by_vix_bucket_subperiods(
    df: pd.DataFrame,
    vrp_col: str = "vrp",
    vix_col: str = "VIX",
    period_edges = ((2010, 2012), (2013, 2015), (2016, 2018), (2019, 2020)),
    period_labels = ("2010–2012", "2013–2015", "2016–2018", "2019–2020"),
    vix_bins = (-np.inf, 15, 20, 30, np.inf),
    vix_labels = ("VIX < 15", "15 ≤ VIX ≤ 20", "20 < VIX ≤ 30", "VIX > 30"),
    figsize=(10, 6),
):

    # --- 1) Period labeling function ---
    def label_period(dt):
        y = dt.year
        for (start, end), label in zip(period_edges, period_labels):
            if start <= y <= end:
                return label
        return None  # outside defined periods

    df = df.copy()

    # --- 2) Add period and VIX bucket columns ---
    df["period"] = df.index.to_series().apply(label_period)
    df["vix_bucket"] = pd.cut(df[vix_col], bins=vix_bins, labels=vix_labels)

    # Drop rows outside defined periods or with missing data
    df = df.dropna(subset=["period", "vix_bucket", vrp_col])

    # --- 3) Make the 2x2 plot ---
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, p in zip(axes, period_labels):
        tmp = (
            df[df["period"] == p]
            .groupby("vix_bucket", observed=False)[vrp_col]
            .agg(["mean", "std", "count"])
        )

        # x positions and values
        x = np.arange(len(tmp))
        means = tmp["mean"].values
        stds = tmp["std"].values  # 1-sigma error bars

        ax.bar(x, means, yerr=stds, capsize=4)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(p)

        ax.set_xticks(x)
        ax.set_xticklabels(tmp.index.astype(str), rotation=20)

    fig.suptitle("Mean VRP by VIX regime across subperiods")
    plt.tight_layout()
    plt.show()

