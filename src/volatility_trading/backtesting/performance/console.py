"""Console-friendly formatting and printing for performance metrics."""

from __future__ import annotations

import pandas as pd

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
    risk_free_rate: float = 0.0,
    alpha: float = 0.01,
) -> PerformanceMetricsBundle:
    """Compute and print a performance report; return metrics for reuse."""
    metrics = compute_performance_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=risk_free_rate,
        alpha=alpha,
    )
    by_contracts = summarize_by_contracts(trades)
    print(format_performance_report(metrics, by_contracts=by_contracts))
    return metrics
