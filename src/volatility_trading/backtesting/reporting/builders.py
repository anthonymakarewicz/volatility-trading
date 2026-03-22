"""Pure builders for backtest report tables and summary metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from volatility_trading.backtesting.options_engine.factor_models import (
    factor_names_from_columns,
    factor_pnl_column,
)
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
_ENTRY_STRESS_DIAGNOSTIC_COLUMNS = [
    "trade_index",
    "entry_date",
    "exit_date",
    "entry_dte",
    "expiry_date",
    "contracts",
    "risk_per_contract",
    "risk_worst_scenario",
    "scenario_name",
    "stress_pnl_per_contract",
    "is_worst_scenario",
    "d_spot",
    "d_volatility",
    "d_risk_reversal",
    "d_rate",
    "dt_years",
]
_STRESS_SCENARIO_SUMMARY_COLUMNS = [
    "scenario_name",
    "times_evaluated",
    "times_worst",
    "worst_frequency",
    "mean_stress_pnl_per_contract",
    "mean_loss_per_contract",
    "max_loss_per_contract",
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


def _normalize_entry_stress_points_value(value: Any) -> list[dict[str, Any]]:
    """Normalize one ``entry_stress_points`` cell to a list-of-dicts payload."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    raw = value
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif isinstance(raw, tuple):
        raw = list(raw)

    if not isinstance(raw, list):
        raise TypeError(
            "entry_stress_points must be a list, tuple, JSON string, or null-like"
        )

    normalized: list[dict[str, Any]] = []
    for point in raw:
        if isinstance(point, dict):
            point_dict = point
        else:
            raise TypeError("each entry stress point must be dict-like")
        normalized.append(
            {
                str(key): _coerce_json_compatible_value(val)
                for key, val in point_dict.items()
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
    if "entry_stress_points" in out.columns:
        out = out.drop(columns=["entry_stress_points"])
    if "trade_legs" not in out.columns:
        return out
    out["trade_legs"] = out["trade_legs"].map(_normalize_trade_legs_value)
    return out


def build_entry_stress_diagnostics_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Explode per-trade entry stress payloads into a diagnostics table."""
    if "entry_stress_points" not in trades.columns:
        return pd.DataFrame(columns=_ENTRY_STRESS_DIAGNOSTIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    for trade_index, trade in trades.reset_index(drop=True).iterrows():
        for point in _normalize_entry_stress_points_value(trade["entry_stress_points"]):
            rows.append(
                {
                    "trade_index": trade_index,
                    "entry_date": _to_timestamp_or_none(trade.get("entry_date")),
                    "exit_date": _to_timestamp_or_none(trade.get("exit_date")),
                    "entry_dte": trade.get("entry_dte"),
                    "expiry_date": _to_timestamp_or_none(trade.get("expiry_date")),
                    "contracts": trade.get("contracts"),
                    "risk_per_contract": trade.get("risk_per_contract"),
                    "risk_worst_scenario": trade.get("risk_worst_scenario"),
                    "scenario_name": point.get("scenario_name"),
                    "stress_pnl_per_contract": point.get("stress_pnl_per_contract"),
                    "is_worst_scenario": bool(point.get("is_worst_scenario", False)),
                    "d_spot": point.get("d_spot"),
                    "d_volatility": point.get("d_volatility"),
                    "d_risk_reversal": point.get("d_risk_reversal"),
                    "d_rate": point.get("d_rate"),
                    "dt_years": point.get("dt_years"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=_ENTRY_STRESS_DIAGNOSTIC_COLUMNS)

    out = pd.DataFrame(rows, columns=_ENTRY_STRESS_DIAGNOSTIC_COLUMNS)
    for col in ["entry_date", "exit_date", "expiry_date"]:
        out[col] = pd.to_datetime(out[col])
    for col in [
        "stress_pnl_per_contract",
        "d_spot",
        "d_volatility",
        "d_risk_reversal",
        "d_rate",
        "dt_years",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_stress_scenario_summary_table(
    entry_stress_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-scenario entry stress diagnostics into a compact summary."""
    if entry_stress_diagnostics.empty:
        return pd.DataFrame(columns=_STRESS_SCENARIO_SUMMARY_COLUMNS)

    required_columns = {
        "scenario_name",
        "stress_pnl_per_contract",
        "is_worst_scenario",
    }
    missing_columns = required_columns.difference(entry_stress_diagnostics.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"entry_stress_diagnostics is missing required columns: {missing}"
        )

    out = entry_stress_diagnostics.copy()
    out["stress_pnl_per_contract"] = pd.to_numeric(
        out["stress_pnl_per_contract"],
        errors="coerce",
    )
    out["is_worst_scenario"] = out["is_worst_scenario"].fillna(False).astype(bool)
    out["loss_per_contract"] = (-out["stress_pnl_per_contract"]).clip(lower=0.0)

    summary = (
        out.groupby("scenario_name", sort=True, dropna=False)
        .agg(
            times_evaluated=("scenario_name", "size"),
            times_worst=("is_worst_scenario", "sum"),
            worst_frequency=("is_worst_scenario", "mean"),
            mean_stress_pnl_per_contract=("stress_pnl_per_contract", "mean"),
            mean_loss_per_contract=("loss_per_contract", "mean"),
            max_loss_per_contract=("loss_per_contract", "max"),
        )
        .reset_index()
    )

    summary["times_evaluated"] = summary["times_evaluated"].astype(int)
    summary["times_worst"] = summary["times_worst"].astype(int)

    return summary.sort_values(
        by=[
            "times_worst",
            "worst_frequency",
            "max_loss_per_contract",
            "scenario_name",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


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
    factor_names = factor_names_from_columns(list(mtm_daily.columns))
    for factor_name in factor_names:
        exposure_cols.append(f"factor_exposure_{factor_name}")
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
    columns = list(_PNL_ATTRIBUTION_COLUMNS)
    factor_pnl_cols = [
        factor_pnl_column(name)
        for name in factor_names_from_columns(list(mtm_daily.columns))
    ]
    columns.extend(column for column in factor_pnl_cols if column not in columns)
    for column in columns:
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
