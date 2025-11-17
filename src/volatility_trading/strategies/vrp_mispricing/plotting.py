import matplotlib.pyplot as plt
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
    axes[1].plot(vrp.index, vrp, label='Variance Risk Premium (IV - RV)', color='tab:green')
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_ylabel('VRP (%)')
    axes[1].set_title('Variance Risk Premium')
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


def plot_vrp_autocorr(vrp, lags=60):
    """Plot ACF and PACF of the Variance Risk Premium."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ACF
    plot_acf(vrp.dropna(), lags=lags, alpha=0.05, ax=axes[0])
    axes[0].set_title(f'VRP Autocorrelation (up to {lags} lags)')
    axes[0].set_xlabel('Lags (trading days)')
    axes[0].set_ylabel('ACF')

    # PACF
    plot_pacf(vrp.dropna(), lags=lags, alpha=0.05, ax=axes[1], method="ywm")
    axes[1].set_title(f'VRP Partial Autocorrelation (up to {lags} lags)')
    axes[1].set_xlabel('Lags (trading days)')
    axes[1].set_ylabel('PACF')

    plt.tight_layout()
    plt.show()
