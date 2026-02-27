"""Tabular output builders for options backtesting runtime records."""

from __future__ import annotations

import pandas as pd

from .records import MtmRecord, TradeRecord


def build_options_backtest_outputs(
    trade_records: list[TradeRecord],
    mtm_records: list[MtmRecord],
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert typed runtime records into canonical ``trades`` and ``mtm`` tables."""
    trades_df = pd.DataFrame([record.to_dict() for record in trade_records])
    if not mtm_records:
        return trades_df, pd.DataFrame()

    mtm_rows = [record.to_dict() for record in mtm_records]
    mtm_agg = pd.DataFrame(mtm_rows).set_index("date").sort_index()
    agg_map = {
        "delta_pnl": "sum",
        "delta": "sum",
        "net_delta": "sum",
        "gamma": "sum",
        "vega": "sum",
        "theta": "sum",
        "hedge_pnl": "sum",
        "S": "first",
        "iv": "first",
    }
    optional_sum_cols = [
        "open_contracts",
        "initial_margin_requirement",
        "maintenance_margin_requirement",
        "contracts_liquidated",
        "financing_pnl",
    ]
    optional_first_cols = [
        "margin_per_contract",
        "margin_excess",
        "margin_deficit",
    ]
    optional_max_cols = ["in_margin_call", "margin_call_days", "forced_liquidation"]
    for col in optional_sum_cols:
        if col in mtm_agg.columns:
            agg_map[col] = "sum"
    for col in optional_first_cols:
        if col in mtm_agg.columns:
            agg_map[col] = "first"
    for col in optional_max_cols:
        if col in mtm_agg.columns:
            agg_map[col] = "max"

    mtm_df = mtm_agg.groupby("date").agg(agg_map)
    mtm_df["equity"] = float(initial_capital) + mtm_df["delta_pnl"].cumsum()
    return trades_df, mtm_df
