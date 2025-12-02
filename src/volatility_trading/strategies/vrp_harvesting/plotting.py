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


import numpy as np
import matplotlib.pyplot as plt


def plot_short_straddle_payoff(
    K: float = 100,
    net_premium: float = 10.0,
    s_min: float = 0.7,
    s_max: float = 1.3,
    n_points: int = 201,
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    S_T = np.linspace(s_min * K, s_max * K, n_points)

    # Long straddle intrinsic payoff (ignoring premium)
    long_intrinsic = np.abs(S_T - K)

    # Short straddle payoff INCLUDING premium:
    # you receive net_premium, then pay intrinsic at expiry
    payoff = net_premium - long_intrinsic

    # Plot payoff line
    ax.plot(S_T, payoff, linewidth=1.5)

    # Shade profit (>=0) and loss (<0)
    ax.fill_between(S_T, payoff, 0, where=(payoff >= 0), color="green", alpha=0.2)
    ax.fill_between(S_T, payoff, 0, where=(payoff < 0), color="red", alpha=0.2)

    # Breakeven points: |S_T - K| = net_premium
    be_low = K - net_premium
    be_high = K + net_premium

    ax.axvline(K, linestyle="--", linewidth=0.8)
    ax.axvline(be_low, linestyle=":", linewidth=0.8)
    ax.axvline(be_high, linestyle=":", linewidth=0.8)
    ax.axhline(0, linewidth=0.8)

    ax.set_title(f"Short ATM Straddle (K={K:.0f}, credit={net_premium:.1f})")
    ax.set_xlabel(r"$S_T$")
    ax.set_ylabel("Payoff at expiry")

    # Optional annotations
    ax.annotate("breakeven", xy=(be_low, 0), xytext=(be_low, net_premium * 0.3),
                ha="center", fontsize=8)
    ax.annotate("breakeven", xy=(be_high, 0), xytext=(be_high, net_premium * 0.3),
                ha="center", fontsize=8)

    return ax


def plot_short_iron_butterfly_payoff(
    K: float = 100,
    dK: float = 10,
    net_premium: float = 3.0,
    s_min: float = 0.7,
    s_max: float = 1.3,
    n_points: int = 201,
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    K1 = K - dK
    K0 = K
    K2 = K + dK

    S_T = np.linspace(s_min * K, s_max * K, n_points)

    # Long call & put payoffs
    C0 = np.maximum(S_T - K0, 0.0)
    C2 = np.maximum(S_T - K2, 0.0)
    P0 = np.maximum(K0 - S_T, 0.0)
    P1 = np.maximum(K1 - S_T, 0.0)

    # Long iron fly intrinsic payoff:
    # long straddle at K0, short wings at K1/K2
    long_iron_intrinsic = C0 + P0 - (C2 + P1)

    # Short iron fly payoff INCLUDING net premium
    payoff = net_premium - long_iron_intrinsic

    # Plot payoff line
    ax.plot(S_T, payoff, linewidth=1.5)

    # Shade profit (>=0) and loss (<0)
    ax.fill_between(S_T, payoff, 0, where=(payoff >= 0), color="green", alpha=0.2)
    ax.fill_between(S_T, payoff, 0, where=(payoff < 0), color="red", alpha=0.2)

    # Mark strikes
    ax.axvline(K1, linestyle=":", linewidth=0.8)
    ax.axvline(K0, linestyle="--", linewidth=0.8)
    ax.axvline(K2, linestyle=":", linewidth=0.8)
    ax.axhline(0, linewidth=0.8)

    ax.set_title(
        f"Short Iron Butterfly (K={K:.0f}±{dK:.0f}, credit={net_premium:.1f})"
    )
    ax.set_xlabel(r"$S_T$")
    ax.set_ylabel("Payoff at expiry")

    return ax