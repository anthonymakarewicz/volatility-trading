"""Pure builders for backtest report tables and summary metrics."""

from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting.performance import compute_performance_metrics
from volatility_trading.backtesting.rates import RateInput

from .schemas import SummaryMetrics

# TODO: Update report builders/writers to persist trade_legs safely (JSON string column for CSV, native list in parquet/json).
# Add tests for multi-expiry trade rows in reporting/performance modules.


def build_summary_metrics(
    trades: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    *,
    risk_free_rate: RateInput = 0.0,
) -> SummaryMetrics:
    """Build headline metrics from trades and daily MTM.

    `risk_free_rate` can be a constant annualized value or a dated series/model.
    """
    metrics = compute_performance_metrics(
        trades=trades,
        mtm_daily=mtm_daily,
        risk_free_rate=risk_free_rate,
    )

    return SummaryMetrics(
        total_return=metrics.returns.total_return,
        cagr=metrics.returns.cagr,
        annualized_volatility=metrics.returns.annualized_volatility,
        sharpe=metrics.returns.sharpe,
        max_drawdown=metrics.drawdown.max_drawdown,
        total_trades=metrics.trades.total_trades,
        win_rate=metrics.trades.win_rate,
        profit_factor=metrics.trades.profit_factor,
    )


def build_equity_and_drawdown_table(
    mtm_daily: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a daily table with strategy and benchmark equity/drawdown."""
    if "equity" not in mtm_daily.columns:
        raise ValueError("mtm_daily must contain an 'equity' column")

    out = pd.DataFrame(index=mtm_daily.index.copy())
    out["equity"] = mtm_daily["equity"].astype(float)
    out["strategy_return"] = out["equity"].pct_change().fillna(0.0)
    out["strategy_drawdown"] = (out["equity"] - out["equity"].cummax()) / out[
        "equity"
    ].cummax()

    if benchmark is None:
        return out

    bm = benchmark.reindex(out.index).ffill()
    if bm.isna().all():
        return out

    first_valid = bm.first_valid_index()
    if first_valid is None:
        return out

    bm = bm.loc[first_valid:]
    out = out.loc[bm.index]
    bm = bm.astype(float)

    if bm.iloc[0] == 0:
        return out

    bm_rebased = (bm / bm.iloc[0]) * out["equity"].iloc[0]
    out["benchmark_rebased"] = bm_rebased
    out["benchmark_return"] = out["benchmark_rebased"].pct_change().fillna(0.0)
    out["benchmark_drawdown"] = (
        out["benchmark_rebased"] - out["benchmark_rebased"].cummax()
    ) / out["benchmark_rebased"].cummax()
    out["relative_equity"] = out["equity"] - out["benchmark_rebased"]
    return out


def build_exposures_daily_table(mtm_daily: pd.DataFrame) -> pd.DataFrame:
    """Extract daily exposure columns for reporting."""
    exposure_cols = [
        "delta",
        "net_delta",
        "gamma",
        "vega",
        "theta",
        "hedge_pnl",
    ]
    present = [col for col in exposure_cols if col in mtm_daily.columns]
    out = mtm_daily[present].copy()
    return out.fillna(0.0)
