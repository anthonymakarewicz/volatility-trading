"""Tabular summaries derived from trade-level backtest outputs."""

from __future__ import annotations

import pandas as pd


def summarize_by_contracts(trades: pd.DataFrame) -> pd.DataFrame:
    """Return contract-size breakdown table for notebook/console display."""
    if trades.empty or "contracts" not in trades.columns or "pnl" not in trades.columns:
        return pd.DataFrame(
            columns=[
                "win_rate",
                "num_trades",
                "total_win_pnl",
                "total_loss_pnl",
                "total_pnl",
            ]
        )

    out = (
        trades.groupby("contracts")
        .agg(
            win_rate=("pnl", lambda x: (x > 0).mean()),
            num_trades=("pnl", "count"),
            total_win_pnl=("pnl", lambda x: x[x > 0].sum()),
            total_loss_pnl=("pnl", lambda x: x[x <= 0].sum()),
            total_pnl=("pnl", "sum"),
        )
        .round(4)
    )
    return out
