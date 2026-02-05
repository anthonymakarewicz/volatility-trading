# volatility_trading/etl/orats/processed/options_chain/_steps/features.py

from __future__ import annotations

import polars as pl


def add_derived_features(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        dte=(pl.col("expiry_date") - pl.col("trade_date")).dt.total_days(),
        moneyness_ks=pl.col("strike") / pl.col("spot_price"),
        call_mid_price=(
            (pl.col("call_bid_price") + pl.col("call_ask_price")) / 2.0
        ),
        put_mid_price=(pl.col("put_bid_price") + pl.col("put_ask_price")) / 2.0,
        call_spread=pl.col("call_ask_price") - pl.col("call_bid_price"),
        put_spread=pl.col("put_ask_price") - pl.col("put_bid_price"),
    )

    lf = lf.with_columns(
        call_rel_spread=(
            pl.when(
                (pl.col("call_mid_price") > 0) & (pl.col("call_spread") >= 0)
            )
            .then(pl.col("call_spread") / pl.col("call_mid_price"))
            .otherwise(None)
        ),
        put_rel_spread=(
            pl.when(
                (pl.col("put_mid_price") > 0) & (pl.col("put_spread") >= 0)
            )
            .then(pl.col("put_spread") / pl.col("put_mid_price"))
            .otherwise(None)
        ),
    )

    return lf