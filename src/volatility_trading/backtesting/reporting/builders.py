"""Pure builders for backtest report tables and summary metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schemas import SummaryMetrics


def _as_optional_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def build_summary_metrics(
    trades: pd.DataFrame,
    mtm_daily: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> SummaryMetrics:
    """Build headline metrics from trades and daily MTM."""
    total_trades = int(len(trades))
    if "equity" not in mtm_daily.columns or mtm_daily.empty:
        return SummaryMetrics(
            total_return=None,
            cagr=None,
            annualized_volatility=None,
            sharpe=None,
            max_drawdown=None,
            total_trades=total_trades,
            win_rate=None,
            profit_factor=None,
        )

    equity = mtm_daily["equity"].astype(float)
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    total_return = (end_val / start_val - 1.0) if start_val > 0 else np.nan

    num_days = (equity.index[-1] - equity.index[0]).days
    num_years = num_days / 365.25 if num_days > 0 else np.nan
    cagr = (
        (end_val / start_val) ** (1.0 / num_years) - 1.0
        if start_val > 0 and num_years and np.isfinite(num_years) and num_years > 0
        else np.nan
    )

    daily_returns = equity.pct_change().dropna()
    annualized_vol = (
        float(daily_returns.std() * np.sqrt(252.0))
        if not daily_returns.empty
        else np.nan
    )
    sharpe = np.nan
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = float(
            ((daily_returns.mean() - risk_free_rate / 252.0) / daily_returns.std())
            * np.sqrt(252.0)
        )

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    win_rate = np.nan
    profit_factor = np.nan
    if total_trades > 0 and "pnl" in trades.columns:
        pnl = trades["pnl"].astype(float)
        win_rate = float((pnl > 0).mean())
        gross_gain = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl <= 0].sum())
        if gross_loss > 0:
            profit_factor = gross_gain / gross_loss

    return SummaryMetrics(
        total_return=_as_optional_float(float(total_return)),
        cagr=_as_optional_float(float(cagr)),
        annualized_volatility=_as_optional_float(float(annualized_vol)),
        sharpe=_as_optional_float(float(sharpe)),
        max_drawdown=_as_optional_float(float(max_drawdown)),
        total_trades=total_trades,
        win_rate=_as_optional_float(float(win_rate)),
        profit_factor=_as_optional_float(float(profit_factor)),
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
