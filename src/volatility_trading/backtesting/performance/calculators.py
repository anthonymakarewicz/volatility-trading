"""Pure performance metric calculators for backtesting outputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from volatility_trading.backtesting.rates import RateInput, coerce_rate_model

from .schemas import (
    DrawdownMetrics,
    PerformanceMetricsBundle,
    ReturnMetrics,
    TailMetrics,
    TradeMetrics,
)


def _index_day_span(index: pd.Index) -> int | None:
    if len(index) < 2:
        return None
    start = index[0]
    end = index[-1]
    if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        return int((end - start).days)
    try:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
    except (TypeError, ValueError):
        return None
    return int((end_ts - start_ts).days)


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _finite_or_none(float(series.mean()))


def _safe_sum(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _finite_or_none(float(series.sum()))


def _to_timestamp_or_none(value: object) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def compute_performance_metrics(
    *,
    trades: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    risk_free_rate: RateInput = 0.0,
    alpha: float = 0.01,
) -> PerformanceMetricsBundle:
    """Compute portfolio, drawdown, tail, and trade metrics from backtest outputs.

    `risk_free_rate` accepts a constant annualized rate, a date-indexed annualized
    rate series, or a custom `RateModel`.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    total_trades = int(len(trades))
    pnl = (
        trades["pnl"].astype(float)
        if not trades.empty and "pnl" in trades.columns
        else pd.Series(dtype=float)
    )
    total_pnl = (
        _safe_sum(mtm_daily["delta_pnl"].astype(float))
        if "delta_pnl" in mtm_daily.columns and not mtm_daily.empty
        else _safe_sum(pnl)
    )

    win_rate = _safe_mean((pnl > 0).astype(float)) if not pnl.empty else None
    avg_win_pnl = _safe_mean(pnl[pnl > 0]) if not pnl.empty else None
    avg_loss_pnl = _safe_mean(pnl[pnl <= 0]) if not pnl.empty else None

    gross_gain = _safe_sum(pnl[pnl > 0]) if not pnl.empty else None
    gross_loss_abs = (
        _finite_or_none(float(-pnl[pnl <= 0].sum())) if not pnl.empty else None
    )
    profit_factor = None
    if gross_gain is not None and gross_loss_abs is not None and gross_loss_abs > 0:
        profit_factor = _finite_or_none(gross_gain / gross_loss_abs)

    trade_frequency = None
    n_days = _index_day_span(mtm_daily.index) if not mtm_daily.empty else None
    if n_days is not None and n_days > 0:
        trade_frequency = _finite_or_none(total_trades / (n_days / 365.25))

    if mtm_daily.empty or "equity" not in mtm_daily.columns:
        return PerformanceMetricsBundle(
            returns=ReturnMetrics(
                total_return=None,
                cagr=None,
                annualized_volatility=None,
                sharpe=None,
            ),
            drawdown=DrawdownMetrics(
                max_drawdown=None,
                average_drawdown=None,
                max_drawdown_duration_days=0,
            ),
            tail=TailMetrics(alpha=alpha, var=None, cvar=None),
            trades=TradeMetrics(
                total_trades=total_trades,
                win_rate=win_rate,
                average_win_pnl=avg_win_pnl,
                average_loss_pnl=avg_loss_pnl,
                total_pnl=total_pnl,
                gross_gain=gross_gain,
                gross_loss=gross_loss_abs,
                profit_factor=profit_factor,
                trade_frequency_per_year=trade_frequency,
            ),
        )

    equity = mtm_daily["equity"].astype(float)
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    total_return = _finite_or_none(end_val / start_val - 1.0) if start_val > 0 else None

    n_days = _index_day_span(equity.index) or 0
    cagr = None
    if start_val > 0 and n_days > 0:
        years = n_days / 365.25
        cagr = _finite_or_none((end_val / start_val) ** (1.0 / years) - 1.0)

    daily_returns = equity.pct_change().dropna()
    annualized_volatility = (
        _finite_or_none(float(daily_returns.std() * np.sqrt(252.0)))
        if not daily_returns.empty
        else None
    )
    sharpe = None
    if not daily_returns.empty and float(daily_returns.std()) > 0:
        rf_model = coerce_rate_model(risk_free_rate)
        rf_daily = pd.Series(
            (
                rf_model.annual_rate(as_of=_to_timestamp_or_none(idx)) / 252.0
                for idx in daily_returns.index
            ),
            index=daily_returns.index,
            dtype=float,
        )
        excess_daily_returns = daily_returns - rf_daily
        sharpe = _finite_or_none(
            float((excess_daily_returns.mean() / daily_returns.std()) * np.sqrt(252.0))
        )

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_drawdown = (
        _finite_or_none(float(drawdown.min())) if not drawdown.empty else None
    )
    avg_drawdown = (
        _finite_or_none(float(drawdown[drawdown < 0].mean()))
        if (drawdown < 0).any()
        else None
    )

    underwater = drawdown != 0
    durations = underwater.groupby((~underwater).cumsum()).cumsum()
    max_drawdown_duration = int(durations.max()) if not durations.empty else 0

    var = None
    cvar = None
    if not daily_returns.empty:
        var_threshold = float(np.quantile(daily_returns, alpha))
        var = _finite_or_none(var_threshold)
        tail_slice = daily_returns[daily_returns <= var_threshold]
        cvar = _safe_mean(tail_slice)

    return PerformanceMetricsBundle(
        returns=ReturnMetrics(
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
        ),
        drawdown=DrawdownMetrics(
            max_drawdown=max_drawdown,
            average_drawdown=avg_drawdown,
            max_drawdown_duration_days=max_drawdown_duration,
        ),
        tail=TailMetrics(alpha=alpha, var=var, cvar=cvar),
        trades=TradeMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            average_win_pnl=avg_win_pnl,
            average_loss_pnl=avg_loss_pnl,
            total_pnl=total_pnl,
            gross_gain=gross_gain,
            gross_loss=gross_loss_abs,
            profit_factor=profit_factor,
            trade_frequency_per_year=trade_frequency,
        ),
    )
