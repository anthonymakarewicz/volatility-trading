"""Pure builders for backtest report tables and summary metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from volatility_trading.backtesting.performance import compute_performance_metrics
from volatility_trading.backtesting.rates import RateInput

from .schemas import SummaryMetrics


def _coerce_json_compatible_value(value: Any) -> Any:
    """Return a JSON-compatible scalar used inside trade-leg payloads."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _normalize_trade_legs_value(value: Any) -> list[dict[str, Any]]:
    """Normalize one ``trade_legs`` cell to a list-of-dicts payload."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    raw = value
    if isinstance(raw, str):
        parsed = json.loads(raw)
        raw = parsed
    elif isinstance(raw, tuple):
        raw = list(raw)

    if not isinstance(raw, list):
        raise TypeError("trade_legs must be a list, tuple, JSON string, or null-like")

    normalized: list[dict[str, Any]] = []
    for leg in raw:
        if hasattr(leg, "to_dict"):
            leg_dict = leg.to_dict()
        elif isinstance(leg, dict):
            leg_dict = leg
        else:
            raise TypeError("each trade leg must be dict-like or expose to_dict()")
        normalized.append(
            {
                str(key): _coerce_json_compatible_value(val)
                for key, val in leg_dict.items()
            }
        )
    return normalized


def build_trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Build canonical reporting trades table with normalized ``trade_legs`` payloads."""
    out = trades.copy()
    if "trade_legs" not in out.columns:
        return out
    out["trade_legs"] = out["trade_legs"].map(_normalize_trade_legs_value)
    return out


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
