"""Pure builders for backtest report tables and summary metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from volatility_trading.backtesting.performance import compute_performance_metrics
from volatility_trading.backtesting.rates import RateInput, coerce_rate_model

from .constants import ROLLING_METRICS_WINDOW
from .schemas import SummaryMetrics

_PNL_ATTRIBUTION_COLUMNS = [
    "delta_pnl",
    "Delta_PnL",
    "Unhedged_Delta_PnL",
    "Gamma_PnL",
    "Vega_PnL",
    "Theta_PnL",
    "Other_PnL",
]
_METRIC_KEYS = [
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
]


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


def _to_timestamp_or_none(value: object) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def _require_equity_frame(mtm_daily: pd.DataFrame) -> pd.DataFrame:
    if "equity" not in mtm_daily.columns:
        raise ValueError("mtm_daily must contain an 'equity' column")
    out = mtm_daily.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _float_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _metric_snapshot(
    *,
    equity: pd.Series,
    risk_free_rate: RateInput,
) -> dict[str, float | None]:
    metrics = compute_performance_metrics(
        trades=pd.DataFrame(),
        mtm_daily=pd.DataFrame({"equity": equity.astype(float)}),
        risk_free_rate=risk_free_rate,
    )
    return {
        "total_return": metrics.returns.total_return,
        "cagr": metrics.returns.cagr,
        "annualized_volatility": metrics.returns.annualized_volatility,
        "sharpe": metrics.returns.sharpe,
        "max_drawdown": metrics.drawdown.max_drawdown,
    }


def _metric_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


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
    mtm_daily = _require_equity_frame(mtm_daily)

    out = pd.DataFrame(index=mtm_daily.index.copy())
    out["equity"] = mtm_daily["equity"].astype(float)
    out["strategy_return"] = out["equity"].pct_change().fillna(0.0)
    out["strategy_drawdown"] = (out["equity"] - out["equity"].cummax()) / out[
        "equity"
    ].cummax()

    if benchmark is None:
        return out

    bm = pd.Series(benchmark).copy()
    bm.index = pd.to_datetime(bm.index)
    bm = bm.sort_index().reindex(out.index).ffill()
    if bm.isna().all():
        return out

    first_valid = bm.first_valid_index()
    if first_valid is None:
        return out

    bm = bm.loc[first_valid:]
    out = out.loc[bm.index]
    bm = bm.astype(float)

    if float(bm.iloc[0]) == 0.0:
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
    out.index = pd.to_datetime(out.index)
    return out.fillna(0.0)


def build_margin_diagnostics_table(mtm_daily: pd.DataFrame) -> pd.DataFrame:
    """Build the persisted daily margin diagnostics table."""
    mtm_daily = _require_equity_frame(mtm_daily)
    out = pd.DataFrame(index=mtm_daily.index.copy())
    out["equity"] = mtm_daily["equity"].astype(float)

    numeric_state_defaults = {
        "open_contracts": 0,
        "margin_per_contract": 0.0,
        "initial_margin_requirement": 0.0,
        "maintenance_margin_requirement": 0.0,
        "margin_call_days": 0,
    }
    boolean_state_defaults = {
        "in_margin_call": False,
    }
    flow_defaults = {
        "forced_liquidation": False,
        "contracts_liquidated": 0,
        "financing_pnl": 0.0,
        "hedge_turnover": 0.0,
        "hedge_trade_count": 0,
    }

    for column, default in numeric_state_defaults.items():
        if column in mtm_daily.columns:
            series = pd.to_numeric(mtm_daily[column], errors="coerce").ffill()
            out[column] = series.fillna(default)
        else:
            out[column] = default

    for column, default in boolean_state_defaults.items():
        if column in mtm_daily.columns:
            series = mtm_daily[column].astype("boolean").ffill().fillna(default)
            out[column] = series.astype(bool)
        else:
            out[column] = default

    if "margin_excess" in mtm_daily.columns:
        out["margin_excess"] = (
            pd.to_numeric(mtm_daily["margin_excess"], errors="coerce")
            .ffill()
            .fillna(out["equity"] - out["maintenance_margin_requirement"])
        )
    else:
        out["margin_excess"] = out["equity"] - out["maintenance_margin_requirement"]

    if "margin_deficit" in mtm_daily.columns:
        out["margin_deficit"] = (
            pd.to_numeric(mtm_daily["margin_deficit"], errors="coerce")
            .ffill()
            .fillna(np.maximum(-out["margin_excess"], 0.0))
        )
    else:
        out["margin_deficit"] = np.maximum(-out["margin_excess"], 0.0)

    for column, default in flow_defaults.items():
        if column in mtm_daily.columns:
            filled = (
                mtm_daily[column].astype("boolean").fillna(default)
                if isinstance(default, bool)
                else mtm_daily[column].fillna(default)
            )
            if isinstance(default, bool):
                out[column] = filled.astype(bool)
            else:
                out[column] = pd.to_numeric(filled, errors="coerce").fillna(default)
        else:
            out[column] = default

    equity = out["equity"].replace(0.0, np.nan)
    out["initial_margin_utilization"] = (
        out["initial_margin_requirement"] / equity
    ).replace([np.inf, -np.inf], np.nan)
    out["maintenance_margin_utilization"] = (
        out["maintenance_margin_requirement"] / equity
    ).replace([np.inf, -np.inf], np.nan)

    out["open_contracts"] = out["open_contracts"].round().astype(int)
    out["margin_call_days"] = out["margin_call_days"].round().astype(int)
    out["contracts_liquidated"] = out["contracts_liquidated"].round().astype(int)
    out["hedge_trade_count"] = out["hedge_trade_count"].round().astype(int)

    ordered_columns = [
        "equity",
        "open_contracts",
        "margin_per_contract",
        "initial_margin_requirement",
        "maintenance_margin_requirement",
        "margin_excess",
        "margin_deficit",
        "initial_margin_utilization",
        "maintenance_margin_utilization",
        "in_margin_call",
        "margin_call_days",
        "forced_liquidation",
        "contracts_liquidated",
        "financing_pnl",
        "hedge_turnover",
        "hedge_trade_count",
    ]
    return out[ordered_columns]


def build_rolling_metrics_table(
    mtm_daily: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    risk_free_rate: RateInput = 0.0,
    window: int = ROLLING_METRICS_WINDOW,
) -> pd.DataFrame:
    """Build rolling return, volatility, Sharpe, and drawdown diagnostics."""
    if window <= 1:
        raise ValueError("window must be > 1")

    equity_and_drawdown = build_equity_and_drawdown_table(
        mtm_daily=mtm_daily,
        benchmark=benchmark,
    )
    out = pd.DataFrame(index=equity_and_drawdown.index.copy())

    equity = equity_and_drawdown["equity"].astype(float)
    strategy_returns = equity.pct_change().fillna(0.0)
    out["strategy_rolling_return"] = equity / equity.shift(window - 1) - 1.0
    out["strategy_rolling_annualized_volatility"] = strategy_returns.rolling(
        window
    ).std() * np.sqrt(252.0)

    rf_model = coerce_rate_model(risk_free_rate)
    rf_daily = pd.Series(
        (
            rf_model.annual_rate(as_of=_to_timestamp_or_none(idx)) / 252.0
            for idx in strategy_returns.index
        ),
        index=strategy_returns.index,
        dtype=float,
    )
    strategy_std = strategy_returns.rolling(window).std()
    out["strategy_rolling_sharpe"] = (
        (strategy_returns - rf_daily).rolling(window).mean() / strategy_std
    ) * np.sqrt(252.0)
    out["strategy_rolling_drawdown"] = equity / equity.rolling(window).max() - 1.0

    if "benchmark_rebased" in equity_and_drawdown.columns:
        benchmark_rebased = equity_and_drawdown["benchmark_rebased"].astype(float)
        benchmark_returns = benchmark_rebased.pct_change().fillna(0.0)
        benchmark_std = benchmark_returns.rolling(window).std()
        out["benchmark_rolling_return"] = (
            benchmark_rebased / benchmark_rebased.shift(window - 1) - 1.0
        )
        out["benchmark_rolling_annualized_volatility"] = benchmark_std * np.sqrt(252.0)
        out["benchmark_rolling_sharpe"] = (
            (benchmark_returns - rf_daily).rolling(window).mean() / benchmark_std
        ) * np.sqrt(252.0)
        out["benchmark_rolling_drawdown"] = (
            benchmark_rebased / benchmark_rebased.rolling(window).max() - 1.0
        )
        out["relative_equity_spread"] = equity - benchmark_rebased

    return out.replace([np.inf, -np.inf], np.nan)


def build_pnl_attribution_daily_table(mtm_daily: pd.DataFrame) -> pd.DataFrame:
    """Build the persisted Greek-attribution daily PnL table."""
    mtm_daily = _require_equity_frame(mtm_daily)
    out = pd.DataFrame(index=mtm_daily.index.copy())
    for column in _PNL_ATTRIBUTION_COLUMNS:
        if column in mtm_daily.columns:
            out[column] = pd.to_numeric(mtm_daily[column], errors="coerce").fillna(0.0)
        else:
            out[column] = 0.0
    return out


def build_benchmark_comparison_payload(
    mtm_daily: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    risk_free_rate: RateInput = 0.0,
) -> dict[str, dict[str, float | None]] | None:
    """Build a compact strategy-vs-benchmark summary payload."""
    if benchmark is None:
        return None

    equity_and_drawdown = build_equity_and_drawdown_table(
        mtm_daily=mtm_daily,
        benchmark=benchmark,
    )
    if "benchmark_rebased" not in equity_and_drawdown.columns:
        return None

    strategy_metrics = _metric_snapshot(
        equity=equity_and_drawdown["equity"],
        risk_free_rate=risk_free_rate,
    )
    benchmark_metrics = _metric_snapshot(
        equity=equity_and_drawdown["benchmark_rebased"],
        risk_free_rate=risk_free_rate,
    )
    relative_metrics = {
        f"{key}_diff": _metric_diff(strategy_metrics[key], benchmark_metrics[key])
        for key in _METRIC_KEYS
    }
    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "relative": relative_metrics,
    }
