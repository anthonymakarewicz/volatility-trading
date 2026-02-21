"""Console-friendly formatting and printing for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_trading.backtesting.rates import RateInput

from .calculators import compute_performance_metrics
from .schemas import PerformanceMetricsBundle
from .tables import summarize_by_contracts


def _fmt_pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def _fmt_usd(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "n/a"


def format_performance_report(
    metrics: PerformanceMetricsBundle,
    *,
    by_contracts: pd.DataFrame | None = None,
) -> str:
    """Format a readable console report from precomputed metrics."""
    alpha_pct = int((1.0 - metrics.tail.alpha) * 100)
    lines = [
        "=" * 40,
        "Overall Performance Metrics",
        "=" * 40,
        f"Sharpe Ratio           : {_fmt_num(metrics.returns.sharpe)}",
        f"CAGR                   : {_fmt_pct(metrics.returns.cagr)}",
        f"Average Drawdown       : {_fmt_pct(metrics.drawdown.average_drawdown)}",
        f"Max Drawdown           : {_fmt_pct(metrics.drawdown.max_drawdown)}",
        f"Max Drawdown Duration  : {metrics.drawdown.max_drawdown_duration_days} days",
        f"Historical VaR ({alpha_pct}%)   : {_fmt_pct(metrics.tail.var)}",
        f"Historical CVaR ({alpha_pct}%)  : {_fmt_pct(metrics.tail.cvar)}",
        f"Total P&L              : {_fmt_usd(metrics.trades.total_pnl)}",
        f"Profit Factor          : {_fmt_num(metrics.trades.profit_factor)}",
        (
            "Trade Frequency (ann.) : "
            f"{_fmt_num(metrics.trades.trade_frequency_per_year, digits=1)} trades/year"
        ),
        f"Total Trades           : {metrics.trades.total_trades}",
        f"Win Rate               : {_fmt_pct(metrics.trades.win_rate)}",
        f"Average Win P&L        : {_fmt_usd(metrics.trades.average_win_pnl)}",
        f"Average Loss P&L       : {_fmt_usd(metrics.trades.average_loss_pnl)}",
        "",
    ]

    if by_contracts is not None and not by_contracts.empty:
        lines.extend(
            [
                "=" * 40,
                "Performance by Contract Size",
                "=" * 40,
                by_contracts.to_string(),
                "",
            ]
        )

    return "\n".join(lines)


def print_performance_report(
    *,
    trades: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    risk_free_rate: RateInput = 0.0,
    alpha: float = 0.01,
) -> PerformanceMetricsBundle:
    """Compute and print a performance report; return metrics for reuse.

    `risk_free_rate` accepts a constant annualized value or a dated series/model.
    """
    metrics = compute_performance_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=risk_free_rate,
        alpha=alpha,
    )
    by_contracts = summarize_by_contracts(trades)
    print(format_performance_report(metrics, by_contracts=by_contracts))
    return metrics


def format_stressed_risk_report(
    *,
    alpha: float,
    base_var: float | None,
    base_cvar: float | None,
    stress_var: float | None,
    stress_cvar: float | None,
) -> str:
    """Format stressed VaR/CVaR diagnostics for console output."""
    alpha_pct = int((1.0 - alpha) * 100)
    lines = [
        f"Base VaR ({alpha_pct}%)     : {_fmt_pct(base_var)}",
        f"Base CVaR ({alpha_pct}%)    : {_fmt_pct(base_cvar)}",
        f"Stress VaR ({alpha_pct}%)   : {_fmt_pct(stress_var)}",
        f"Stress CVaR ({alpha_pct}%)  : {_fmt_pct(stress_cvar)}",
    ]
    return "\n".join(lines)


def print_stressed_risk_metrics(
    *,
    stressed_mtm: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    alpha: float = 0.01,
) -> dict[str, float | None]:
    """Print stressed and unstressed historical VaR/CVaR diagnostics."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if "equity" not in mtm_daily.columns:
        raise ValueError("mtm_daily must contain an 'equity' column")
    if stressed_mtm.empty:
        raise ValueError("stressed_mtm must not be empty")

    equity = mtm_daily["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)

    shock_pct = (
        stressed_mtm.reindex(equity.index).astype(float).div(equity.shift(1), axis=0)
    )
    shock_pct = shock_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_returns = shock_pct.add(returns, axis=0)
    worst_stressed_returns = total_returns.min(axis=1)

    base_var_raw = returns.quantile(alpha)
    base_cvar_raw = returns.loc[returns <= base_var_raw].mean()
    stress_var_raw = worst_stressed_returns.quantile(alpha)
    stress_cvar_raw = worst_stressed_returns.loc[
        worst_stressed_returns <= stress_var_raw
    ].mean()

    metrics = {
        "base_var": float(base_var_raw) if pd.notna(base_var_raw) else None,
        "base_cvar": float(base_cvar_raw) if pd.notna(base_cvar_raw) else None,
        "stress_var": float(stress_var_raw) if pd.notna(stress_var_raw) else None,
        "stress_cvar": float(stress_cvar_raw) if pd.notna(stress_cvar_raw) else None,
    }
    print(format_stressed_risk_report(alpha=alpha, **metrics))
    return metrics
