
import matplotlib.pyplot as plt
import numpy as np
from config.constants import OPTION_TYPES


def plot_iv_smiles(iv_smiles, ticker):
    plt.figure(figsize=(12, 6))

    for dte, iv_smile in iv_smiles.items():
        if iv_smile is not None and not iv_smile.empty:
            plt.scatter(iv_smile.index, iv_smile.values, marker='o', label=int(dte))

    plt.xlabel('Strike', fontsize=14)
    plt.ylabel('Implied Volatility', fontsize=14)
    plt.title(f'Implied Volatility Smiles for {ticker} Options', fontsize=16)
    plt.legend(title="Days to Expiry", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_volume_filter(options, log_volumes):
    plt.figure(figsize=(10, 5))

    for opt_type, color in zip(OPTION_TYPES, ["blue", "orange"]):
        subset = options[options["option_type"] == opt_type]
        log_volumes = (subset["volume"] + 1).apply(np.log)
        log_volumes.hist(bins=100, alpha=0.5, label=f"{opt_type} options", color=color)

    plt.axvline(np.log(2), color='r', linestyle='--', label='Volume ≥ 1')
    plt.title("Option Volume Distribution (log scale)", fontsize=16)
    plt.xlabel("log(volume + 1)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bid_ask_filter(options):
    plt.figure(figsize=(10, 6))

    for opt_type, color in zip(OPTION_TYPES, ["blue", "orange"]):
        subset = options[options["option_type"] == opt_type]
        log_rel_spread = (subset["rel_spread"] + 1e-6).apply(np.log)
        log_rel_spread.hist(bins=100, alpha=0.5, label=f"{opt_type} options", color=color)

    plt.axvline(np.log(0.25), color='red', linestyle='--', label="25% max spread")
    plt.xlabel("log(Relative Bid-Ask Spread)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of Relative Bid-Ask Spread (Calls vs Puts)", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_moneyness_filter(avg_vol):
    plt.figure(figsize=(12, 6))

    for option_type in OPTION_TYPES:
        subset = avg_vol[avg_vol["option_type"] == option_type]
        plt.plot(
            subset["moneyness_bin"].astype(str),
            subset["volume"],
            label=f"{option_type} volume"
        )

    plt.axvline("(0.8, 0.85]", color='red', linestyle='--', label="0.8 Moneyness")
    plt.axvline("(1.2, 1.25]", color='red', linestyle='--', label="1.2 Moneyness")
    plt.xticks(rotation=45)
    plt.xlabel("Moneyness (strike / underlying)", fontsize=14)
    plt.ylabel("Average Volume", fontsize=14)
    plt.title("Average Option Volume by Moneyness", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_synthetic_ivs(synthetic_skew):
    plt.figure(figsize=(12, 6))
    plt.plot(synthetic_skew.index, synthetic_skew['iv_put_30'], label='25Δ Put IV')
    plt.plot(synthetic_skew.index, synthetic_skew['iv_call_30'], label='25Δ Call IV')
    plt.plot(synthetic_skew.index, synthetic_skew['iv_atm_30'], label='ATM IV')

    plt.title('Synthetic 30-DTE Implied Volatilities', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Implied Volatility', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_norm_abs_skew(synthetic_skew):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(synthetic_skew['skew_norm'], color="red", label="Normalized Skew")
    ax1.set_ylabel("Normalized Skew", color="red", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(synthetic_skew['skew_abs'], color="navy", label="Absolute Skew")
    ax2.set_ylabel("Absolute Skew", color="navy", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="navy")

    ax1.set_title("Absolute vs Normalized 30-DTE 25Δ Skew", fontsize=16)
    ax1.set_xlabel("Date", fontsize=14)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_skew_vs_spy(synthetic_skew, spy):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- First subplot: SPY vs Absolute Skew ---
    ax1 = axes[0]
    ax1.plot(spy["Close"], color="purple", label="SPY")
    ax1.set_ylabel("SPY", color="purple", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="purple")
    ax1.set_title("SPY vs 30-DTE 25Δ Absolute Skew", fontsize=14)

    ax1b = ax1.twinx()
    ax1b.plot(synthetic_skew["skew_abs"], color="blue", label="Absolute Skew")
    ax1b.set_ylabel("Abs Skew", color="blue", fontsize=12)
    ax1b.tick_params(axis='y', labelcolor="blue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc="upper left")

    # --- Second subplot: SPY vs Normalized Skew ---
    ax2 = axes[1]
    ax2.plot(spy["Close"], color="purple", label="SPY")
    ax2.set_ylabel("SPY", color="purple", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="purple")
    ax2.set_title("SPY vs 30-DTE 25Δ Normalized Skew", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)

    ax2b = ax2.twinx()
    ax2b.plot(synthetic_skew["skew_norm"], color="red", label="Normalized Skew")
    ax2b.set_ylabel("Norm Skew", color="red", fontsize=12)
    ax2b.tick_params(axis='y', labelcolor="red")

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_risk_reversal_payoff(spot_price=100, strike_put=95, strike_call=105, premium_put=3, premium_call=2):
    price_range = np.linspace(80, 120, 500)

    # Compute payoff
    payoff_call = np.maximum(price_range - strike_call, 0) - premium_call
    payoff_put = -np.maximum(strike_put - price_range, 0) + premium_put
    risk_reversal_pnl = payoff_call + payoff_put

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(price_range, risk_reversal_pnl, label='Risk Reversal Payoff (Buy Call, Sell Put)', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(spot_price, color='black', linestyle=':', label='Spot Price')
    plt.xlabel('Underlying Price at Expiry', fontsize=14)
    plt.ylabel('P&L', fontsize=14)
    plt.title('Risk Reversal Payoff Diagram', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()