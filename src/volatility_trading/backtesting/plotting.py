import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec


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