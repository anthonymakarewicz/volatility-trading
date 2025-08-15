import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
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
    fig, ax1 = plt.subplots(figsize=(14, 5))

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
    plt.figure(figsize=(12, 6))
    plt.plot(price_range, risk_reversal_pnl, label='Risk Reversal Payoff (Buy Call, Sell Put)', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(spot_price, color='black', linestyle=':', label='Spot Price')
    plt.xlabel('Underlying Price at Expiry', fontsize=14)
    plt.ylabel('P&L', fontsize=14)
    plt.title('Risk Reversal Payoff Diagram', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_skew_signals(skew, signals, lower_threshold, upper_threshold, title="Skew with Entry/Exit Signals"):
    plt.figure(figsize=(14, 5))
    plt.plot(skew, label='Skew', color='blue')
    
    plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
    plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
    
    # Entry/Exit markers
    plt.scatter(signals[signals['long']].index, skew[signals['long']], color='green', marker='^', label='Long Signal')
    plt.scatter(signals[signals['short']].index, skew[signals['short']], color='red', marker='v', label='Short Signal')
    plt.scatter(signals[signals['exit']].index, skew[signals['exit']], color='purple', marker='D', label='Exit Signal')

    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Skew", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_zscore_signals(z_score, signals, entry_threshold, exit_threshold,
                        title="Skew Z-Score with Entry/Exit Signals"):
    plt.figure(figsize=(14, 5))

    plt.plot(z_score, label='Z-Score', color='blue')
    plt.axhline(entry_threshold, color='red', linestyle='--', label='Entry Threshold')
    plt.axhline(-entry_threshold, color='red', linestyle='--')
    plt.axhline(exit_threshold, color='green', linestyle='--', label='Exit Threshold')
    plt.axhline(-exit_threshold, color='green', linestyle='--')

    plt.scatter(signals[signals['long']].index, z_score[signals['long']], 
                color='green', marker='^', label='Long Entry')
    plt.scatter(signals[signals['short']].index, z_score[signals['short']], 
                color='red', marker='v', label='Short Entry')
    plt.scatter(signals[signals['exit']].index, z_score[signals['exit']], 
                color='purple', marker='D', label='Exit')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Z-Score", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_skew_vs_zscore(synthetic_skew):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    zscore = synthetic_skew["skew_zscore"].dropna()
    skew = synthetic_skew["skew_abs"].loc[zscore.index]

    ax1.plot(skew, label="Skew", color="blue")
    ax1.set_ylabel("Skew", color="blue", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(zscore, label="Z-Score", color="orange")
    ax2.set_ylabel("Z-Score", color="orange", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("Raw Skew and Z-Score of Skew", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_boll_bands(synthetic_skew, signals):
    window = 60            # e.g. 60-day rolling
    k_entry = 1.5          # entry at ±1.5σ
    k_exit  = 0.5          # exit at ±0.5σ

    # rolling stats
    m = synthetic_skew['skew_abs'].rolling(window).mean().dropna()
    s = synthetic_skew['skew_abs'].rolling(window).std().dropna()
    skew = synthetic_skew['skew_abs'].loc[m.index]

    # bands
    upper_entry = m + k_entry * s
    lower_entry = m - k_entry * s
    upper_exit  = m + k_exit  * s
    lower_exit  = m - k_exit  * s

    plt.figure(figsize=(14, 5))
    plt.plot(skew, label='Skew', color='blue')
    plt.plot(m, label='Rolling Mean', color='black')
    plt.plot(upper_entry, '--', label=f'{k_entry}σ Entry', color='red')
    plt.plot(lower_exit,  ':', label=f'{k_exit}σ Exit',  color='green')
    plt.plot(lower_entry, '--', color="red")
    plt.plot(upper_exit,  ':', color="green")

    # overlay your signals
    plt.scatter(signals[signals.long].index,
                synthetic_skew.loc[signals.long, 'skew_abs'],
                marker='^', color='green', label='Long Entry')
    plt.scatter(signals[signals.short].index,
                synthetic_skew.loc[signals.short, 'skew_abs'],
                marker='v', color='red',   label='Short Entry')
    plt.scatter(signals[signals.exit].index,
                synthetic_skew.loc[signals.exit, 'skew_abs'],
                marker='D', color='purple',label='Exit')

    plt.title("Skew with Bollinger-Style Bands and Signals", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Skew", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zscore_signals_with_vix(z_score, signals, vix, entry_threshold, exit_threshold, title, vix_filter=20):
    fig, ax = plt.subplots(figsize=(14, 5))

    z_score = z_score.copy()
    signals = signals.copy()
    vix = vix.copy()

    common_dates = z_score.index.intersection(vix.index).intersection(signals.index)
    z_score = z_score.loc[common_dates]
    signals = signals.loc[common_dates]
    vix = vix.loc[common_dates]

    # Plot Skew (Left Axis)
    ax.plot(z_score, label="Skew Z-Score", color="blue")

    ax.axhline(entry_threshold, color='red', linestyle='--')
    ax.axhline(-entry_threshold, color='red', linestyle='--')
    ax.axhline(exit_threshold, color='green', linestyle='--')
    ax.axhline(-exit_threshold, color='green', linestyle='--')

    # Entry and Exit markers
    ax.scatter(signals[signals['long']].index, 
               z_score[signals['long']], 
               color='green', marker='^', s=50, label='Long Entry')
    ax.scatter(signals[signals['short']].index, 
               z_score[signals['short']], 
               color='red', marker='v', s=50, label='Short Entry')
    ax.scatter(signals[signals['exit']].index, 
               z_score[signals['exit']], 
               color='purple', marker='D', s=50, label='Exit')

    ax.set_ylabel("Z-Score", color="blue")
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=16)

    # Create masks
    high_vix = (vix > vix_filter)
    low_vix = ~high_vix

    def shade_regions(mask, color, alpha, label):
        start = None
        for date, active in mask.items():
            if active and start is None:
                start = date
            elif not active and start is not None:
                ax.axvspan(start, date, color=color, alpha=alpha,
                           label=label if label not in ax.get_legend_handles_labels()[1] else None)
                start = None
        if start is not None:
            ax.axvspan(start, mask.index[-1], color=color, alpha=alpha,
                       label=label if label not in ax.get_legend_handles_labels()[1] else None)

    # Shade regions
    shade_regions(high_vix, 'red', 0.15, f'VIX > {vix_filter}')
    shade_regions(low_vix, 'green', 0.1, f'VIX ≤ {vix_filter}')

    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_vix(vix, vix_threshold=20):
    plt.figure(figsize=(12, 6))
    plt.plot(vix, label="VIX", color="blue")
    plt.axhline(vix_threshold, color="red", linestyle="--", label=f"VIX = {vix_threshold}")

    vix = vix.copy()
    #vix = vix.iloc[:, 0]
    # Fill above threshold
    above = vix > vix_threshold
    
    plt.fill_between(
        vix.index,
        vix_threshold,
        np.where(above, vix, np.nan),  # fill only where VIX > threshold
        color='red',
        alpha=0.2,
    )
    plt.fill_between(
        vix.index,
        vix_threshold,
        np.where(~above, vix, np.nan),  # fill only where VIX > threshold
        color='green',
        alpha=0.2,
    )

    plt.title("VIX Time Series with Regime Threshold", fontsize=16)
    plt.ylabel("VIX", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.legend()
    plt.show()


def plot_ivp(ivp, ivp_lower_threshold=30, ivp_higher_threshold=70):
    ivp = ivp.copy()
    ivp = ivp.dropna()

    ivp_lower_threshold /= 100
    ivp_higher_threshold /= 100

    # Fill above threshold
    plt.figure(figsize=(12, 6))
    plt.plot(ivp, label="IVP", color="blue")
    plt.axhline(ivp_lower_threshold, color="red", linestyle="--")
    plt.axhline(ivp_higher_threshold, color="red", linestyle="--")
    
    plt.fill_between(ivp.index, ivp_higher_threshold, ivp, where=(ivp >= ivp_higher_threshold), 
                     color='red', alpha=0.2, interpolate=True, label=f"Extreme regime (IVP <= {ivp_lower_threshold}% & IVP >= {ivp_higher_threshold}%)")
    
    plt.fill_between(ivp.index, ivp_lower_threshold, ivp, where=(ivp <= ivp_lower_threshold),
                     color='red', alpha=0.2, interpolate=True)

    plt.fill_between(ivp.index, ivp_lower_threshold, ivp_higher_threshold,
                     color='green', alpha=0.2, interpolate=True, label=f"Normal regime ({ivp_lower_threshold}% ≤ IVP < {ivp_higher_threshold}%)")

    plt.title("IV Percentile (IVP) Regimes", fontsize=16)
    plt.ylabel("IVP", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zscore_signals_with_ivp(
    z_score, signals, ivp, entry_threshold, exit_threshold, 
    ivp_lower_threshold, ivp_higher_threshold):
    
    z_score = z_score.copy()
    ivp = ivp.copy()
    signals = signals.copy()

    ivp = ivp.dropna()
    z_score = z_score.dropna()

    common_dates = z_score.index.intersection(ivp.index).intersection(signals.index)
    z_score = z_score.loc[common_dates]
    signals = signals.loc[common_dates]
    ivp = ivp.loc[common_dates]

    fig, ax = plt.subplots(figsize=(14, 5))
    # Plot Skew (Left Axis)
    ax.plot(z_score, label="Skew Z-Score", color="blue")

    ax.axhline(entry_threshold, color='red', linestyle='--')
    ax.axhline(-entry_threshold, color='red', linestyle='--')
    # Exit thresholds
    ax.axhline(exit_threshold, color='green', linestyle='--')
    ax.axhline(-exit_threshold, color='green', linestyle='--')

    # Entry and Exit markers
    ax.scatter(signals[signals['long']].index, 
               z_score[signals['long']], 
               color='green', marker='^', s=50)
    ax.scatter(signals[signals['short']].index, 
               z_score[signals['short']], 
               color='red', marker='v', s=50)
    ax.scatter(signals[signals['exit']].index, 
               z_score[signals['exit']], 
               color='purple', marker='D', s=50)

    # Build boolean masks
    normal_band   = (ivp_lower_threshold/100 <= ivp) & (ivp <= ivp_higher_threshold/100)
    extreme_band  = ~normal_band

    # Helper to shade a mask with a given color
    def shade(mask, color, alpha, label):
        start = None
        for date, val in mask.items():
            if val and start is None:
                start = date
            if start is not None and (not val):
                ax.axvspan(start, date, color=color, alpha=alpha, 
                           label=label if label not in ax.get_legend_handles_labels()[1] else None)
                start = None
        if start is not None:
            ax.axvspan(start, mask.index[-1], color=color, alpha=alpha,
                       label=label if label not in ax.get_legend_handles_labels()[1] else None)

    # Shade outside/mid in orange, panic in red
    shade(normal_band,   'green', 0.15, f'IVP inside [{ivp_lower_threshold}%,{ivp_higher_threshold}%]')
    shade(extreme_band,   'red',    0.15, f'IVP >= outside [{ivp_lower_threshold}%,{ivp_higher_threshold}%]')

    ax.grid(True)
    ax.set_ylabel("Z-Score", color="blue")
    ax.set_xlabel("Date")
    ax.set_title("Skew with Signals and IVP Filter", fontsize=16)
    ax.legend(loc="lower left")
    plt.tight_layout()

    plt.show()


def plot_skew_percentile(skew_perc, lower_threshold=20, upper_threshold=80):
    skew_perc = skew_perc.copy().dropna()

    lower_threshold /= 100
    upper_threshold /= 100

    plt.figure(figsize=(12, 6))
    plt.plot(skew_perc, label="Skew Percentile", color="blue")
    plt.axhline(lower_threshold, color="red", linestyle="--")
    plt.axhline(upper_threshold, color="red", linestyle="--")

    plt.fill_between(
        skew_perc.index, upper_threshold, skew_perc,
        where=(skew_perc >= upper_threshold),
        color='red', alpha=0.2, interpolate=True,
        label=f"Extreme regime (≤ {lower_threshold*100:.0f}% or ≥ {upper_threshold*100:.0f}%)"
    )

    plt.fill_between(
        skew_perc.index, lower_threshold, skew_perc,
        where=(skew_perc <= lower_threshold),
        color='red', alpha=0.2, interpolate=True,
    )

    plt.fill_between(
        skew_perc.index, lower_threshold, upper_threshold,
        color='green', alpha=0.2, interpolate=True,
        label=f"Normal regime ({lower_threshold*100:.0f}% – {upper_threshold*100:.0f}%)"
    )

    plt.title("Skew Percentile Regimes", fontsize=16)
    plt.ylabel("Skew Percentile", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_zscore_signals_with_skew_percentile(
    z_score, signals, skew_perc, entry_threshold, exit_threshold, 
    lower_threshold=20, upper_threshold=80, title="Z-Score and Skew Percentile Regimes"):

    lower_threshold /= 100
    upper_threshold /= 100

    z_score = z_score.copy().dropna()
    skew_perc = skew_perc.copy().dropna()
    signals = signals.copy()

    common_dates = z_score.index.intersection(skew_perc.index).intersection(signals.index)
    z_score = z_score.loc[common_dates]
    signals = signals.loc[common_dates]
    skew_perc = skew_perc.loc[common_dates]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(z_score, label="Skew Z-Score", color="blue")

    ax.axhline(entry_threshold, color='red', linestyle='--')
    ax.axhline(-entry_threshold, color='red', linestyle='--')
    ax.axhline(exit_threshold, color='green', linestyle='--')
    ax.axhline(-exit_threshold, color='green', linestyle='--')

    # Entry/Exit Markers
    ax.scatter(signals[signals['long']].index,
               z_score[signals['long']],
               color='green', marker='^', s=50)
    ax.scatter(signals[signals['short']].index,
               z_score[signals['short']],
               color='red', marker='v', s=50)
    ax.scatter(signals[signals['exit']].index,
               z_score[signals['exit']],
               color='purple', marker='D', s=50, label='Exit')

    # Define extreme and normal regimes
    extreme_band = (skew_perc <= lower_threshold) | (skew_perc >= upper_threshold)
    normal_band  = ~extreme_band

    def shade(mask, color, alpha, label):
        start = None
        for date, val in mask.items():
            if val and start is None:
                start = date
            elif not val and start is not None:
                ax.axvspan(start, date, color=color, alpha=alpha,
                           label=label if label not in ax.get_legend_handles_labels()[1] else None)
                start = None
        if start is not None:
            ax.axvspan(start, mask.index[-1], color=color, alpha=alpha,
                       label=label if label not in ax.get_legend_handles_labels()[1] else None)

    shade(extreme_band, 'green', 0.15, f'Skew Percentile ≤ {lower_threshold*100:.0f}% or ≥ {upper_threshold*100:.0f}%')
    shade(normal_band, 'red',   0.15, f'Skew Percentile in ({lower_threshold*100:.0f}%, {upper_threshold*100:.0f}%)')
    ax.set_ylabel("Z-Score", color="blue")
    ax.set_xlabel("Date")
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_eq_curve(mtm, sp500):
    # Align both series on the same date index
    sp500 = sp500.loc[mtm.index.min():mtm.index.max()]
    sp500 = sp500.ffill()  # handle missing values

    # Rebase S&P 500 to start at the same value as the mtm curve
    sp500_rebased = (sp500 / sp500.iloc[0]) * mtm.equity.iloc[0]

    # Calculate drawdown
    peak = mtm.equity.cummax()
    drawdown = (mtm.equity - peak) / peak

    # Setup subplots: 2x1 layout
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Equity curve vs S&P 500
    axes[0].plot(mtm.index, mtm.equity, label="Equity Curve", color="blue")
    axes[0].plot(sp500_rebased.index, sp500_rebased, label="S&P 500 (Rebased)", color="orange")
    axes[0].set_title("Equity Curve vs. S&P 500")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Drawdown
    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def print_perf_metrics(trades, mtm, risk_free_rate=0.00, alpha=0.01):
    total_trades = len(trades)
    win_rate = (trades.pnl > 0).mean()
    avg_pnl_win = trades.loc[trades.pnl > 0, "pnl"].mean()
    avg_pnl_lose = trades.loc[trades.pnl <= 0, "pnl"].mean()
    total_pnl = mtm["delta_pnl"].sum()

    # Profit factor = gross gains / gross losses (absolute)
    gross_gain = trades.loc[trades.pnl > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades.pnl <= 0, "pnl"].sum()
    profit_factor = gross_gain / gross_loss if gross_loss != 0 else np.nan

    # Trade frequency (annualized)
    n_days = (mtm.index[-1] - mtm.index[0]).days
    trade_freq = total_trades / (n_days / 365.25) if n_days > 0 else np.nan

    summary_by_contracts = trades.groupby("contracts").agg(
        win_rate=('pnl', lambda x: (x > 0).mean()),
        num_trades=('pnl', 'count'),
        total_win_pnl=('pnl', lambda x: x[x > 0].sum()),
        total_loss_pnl=('pnl', lambda x: x[x <= 0].sum()),
        total_pnl=('pnl', 'sum')
    ).round(2)

    # --- Daily returns for Sharpe ---
    daily_returns = mtm.equity.pct_change().dropna()
    sharpe_ratio = ((daily_returns.mean() - risk_free_rate / 252) / daily_returns.std()) * np.sqrt(252)

    # --- CAGR ---
    start_val = mtm.equity.iloc[0]
    end_val = mtm.equity.iloc[-1]
    num_years = (mtm.index[-1] - mtm.index[0]).days / 365.25
    cagr = (end_val / start_val) ** (1 / num_years) - 1 if num_years > 0 else np.nan

    # --- Drawdown metrics ---
    cumulative = mtm.equity
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean()


    underwater = drawdown != 0
    durations = (underwater.groupby((~underwater).cumsum()).cumsum())
    max_drawdown_duration = durations.max() if not durations.empty else 0

    var = np.quantile(daily_returns, alpha)
    cvar = daily_returns[daily_returns <= var].mean()

    print("=" * 40)
    print("🔍 Overall Performance Metrics")
    print("=" * 40)
    print(f"Sharpe Ratio           : {sharpe_ratio:.2f}")
    print(f"CAGR                   : {cagr:.2%}")
    print(f"Average Drawdown       : {avg_drawdown:.2%}")
    print(f"Max Drawdown           : {max_drawdown:.2%}")
    print(f"Max Drawdown Duration  : {max_drawdown_duration} days")
    print(f"Historical VaR ({int((1-alpha)*100)}%)   : {var:.2%}")
    print(f"Historical CVaR ({int((1-alpha)*100)}%)  : {cvar:.2%}")
    print(f"Total P&L              : ${total_pnl:,.2f}")
    print(f"Profit Factor          : {profit_factor:.2f}")
    print(f"Trade Frequency (ann.) : {trade_freq:.1f} trades/year")
    print(f"Total Trades           : {total_trades}")
    print(f"Win Rate               : {win_rate:.2%}")
    print(f"Average Win P&L        : ${avg_pnl_win:,.2f}")
    print(f"Average Loss P&L       : ${avg_pnl_lose:,.2f}")
    print()

    print("=" * 40)
    print("📊 Performance by Contract Size")
    print("=" * 40)
    print(summary_by_contracts.to_string())
    print()


def plot_full_performance(sp500, mtm_daily):
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # 1) align & rebase
    equity     = mtm_daily['equity']
    spx        = sp500.reindex(equity.index).ffill()
    spx_rebase = spx / spx.iloc[0] * equity.iloc[0]

    # 2) drawdowns
    strat_dd = (equity - equity.cummax()) / equity.cummax()
    spx_dd   = (spx_rebase - spx_rebase.cummax()) / spx_rebase.cummax()

    # 3) set up 4×2 grid
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs  = gridspec.GridSpec(
        4, 2,
        height_ratios=[2, 1, 1, 1],
        hspace=0.4, wspace=0.3
    )

    # ─── row 0: Equity vs SPX ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(equity.index, equity,      color='tab:blue',   label='Equity Curve')
    ax0.plot(spx_rebase.index, spx_rebase, color='tab:orange', label='S&P 500 (rebased)')
    ax0.set_title("Equity vs. S&P 500")
    ax0.set_ylabel("Portfolio Value")
    ax0.legend(loc='upper left')
    ax0.grid(True)

    # ─── row 1: Drawdown ─────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    ax1.fill_between(strat_dd.index, strat_dd, 0,   color='tab:blue',   alpha=0.3, label='Strategy DD')
    ax1.fill_between(spx_dd.index,   spx_dd,   0,   color='tab:orange', alpha=0.3, label='S&P 500 DD')
    ax1.set_title("Drawdown")
    ax1.set_ylabel("Drawdown (%)")
    ax1.legend(loc='lower left')
    ax1.grid(True)

    # ─── rows 2–3: Greeks in 2×2 ─────────────────────────
    greek_cols   = ['net_delta','gamma','vega','theta']
    greek_titles = ['Total Δ Exposure','Total Γ Exposure',
                    'Total ν Exposure','Total Θ Exposure']
    colors       = ['red','orange','green','blue']

    for i, (col, title, color) in enumerate(zip(greek_cols, greek_titles, colors)):
        row    = 2 + (i // 2)
        column = i % 2
        ax = fig.add_subplot(gs[row, column])
        ax.plot(mtm_daily.index, mtm_daily[col], color=color)
        ax.set_title(title)
        ax.set_ylabel(col.capitalize())
        if row == 3:
            ax.set_xlabel("Date")
        ax.grid(True)

    plt.show()


def plot_pnl_attribution(daily_mtm):
    cumu = pd.DataFrame(index=daily_mtm.index)
    cumu['Total P&L'] = daily_mtm['equity'] - daily_mtm['equity'].iloc[0]
    for greek in ['Delta_PnL','Gamma_PnL','Vega_PnL','Theta_PnL','Other_PnL']:
        cumu[greek] = daily_mtm[greek].cumsum()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(cumu.index, cumu['Total P&L'], label='Total P&L')
    for col in cumu.columns.drop('Total P&L'):
        ax.plot(cumu.index, cumu[col], label=col)
    ax.set_title('Cumulative P&L Attribution: Total vs Greek Contributions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (USD)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_stressed_pnl(stressed_mtm, daily_mtm, scenarios):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(daily_mtm['equity'], label='Actual Equity')
    scenarios = ["PnL_" + scenario_name for scenario_name in scenarios.keys()]
    for name in scenarios:
        ax.plot(daily_mtm['equity'] + stressed_mtm[name].cumsum(), label=name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (USD)')
    ax.set_title('Equity Curve vs. Stressed Equity Curves')
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_stressed_risk_metrics(stressed_mtm, daily_mtm, alpha=0.01):
    daily_mtm = daily_mtm.copy()

    # 1) Compute your actual daily returns
    returns = daily_mtm['equity'].pct_change().fillna(0.0)

    # 2) Build the scenario shock PnLs (as % of prior equity)
    shock_pct = stressed_mtm.div(daily_mtm['equity'].shift(1), axis=0)

    # 3) Build total stressed returns = actual + shock each scenario
    total_ret = pd.DataFrame({
        name: returns + shock_pct[name]
        for name in shock_pct.columns
    })

    # 4) Pick the worst‐of‐scenarios daily stressed return
    total_ret['worst_stressed_ret'] = total_ret.min(axis=1)

    # 5) Compute VaR and ES on the two series
    # Base (actual) VaR/ES
    base_var = returns.quantile(alpha)
    base_es  = returns.loc[returns <= base_var].mean()

    # Stressed VaR/ES
    stress_var = total_ret['worst_stressed_ret'].quantile(alpha)
    stress_es  = total_ret.loc[
        total_ret['worst_stressed_ret'] <= stress_var,
        'worst_stressed_ret'
    ].mean()

    print(f"Base VaR ({int((1-alpha)*100)}%)     : {base_var:.2%}")
    print(f"Base CVaR ({int((1-alpha)*100)}%)    : {base_es:.2%}")
    print(f"Stress VaR ({int((1-alpha)*100)}%)   : {stress_var:.2%}")
    print(f"Stress CVaR ({int((1-alpha)*100)}%)  : {stress_es:.2%}")


def plot_smiles(iv_surf, dtes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    titles = [f"{T}-day DTE" for T in dtes]

    for ax, dte, title in zip(axes, dtes, titles):
        iv_surf.plot_smile(
            T=dte/252,
            ax=ax
        )
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_vrp(iv_atm, rv, vrp):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot IV and RV
    axes[0].plot(iv_atm.index, iv_atm['iv_atm'], label='30D ATM IV', color='tab:blue')
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

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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