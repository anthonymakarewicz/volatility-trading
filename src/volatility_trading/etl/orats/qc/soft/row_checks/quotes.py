from __future__ import annotations

import polars as pl


def flag_locked_market(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    out_col: str = "locked_market_violation",
) -> pl.DataFrame:
    """Locked market: bid == ask > 0 (liquidity diagnostic)."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return (
        df_long
        .filter(pl.col("option_type") == option_type)
        .with_columns(
            ((pl.col(bid_col) == pl.col(ask_col)) & (pl.col(bid_col) > 0))
            .fill_null(False)
            .alias(out_col)
        )
    )


def flag_one_sided_quotes(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    out_col: str = "one_sided_quote_violation",
) -> pl.DataFrame:
    """One-sided quote: bid == 0 and ask > 0."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return (
        df_long
        .filter(pl.col("option_type") == option_type)
        .with_columns(
            ((pl.col(bid_col) == 0) & (pl.col(ask_col) > 0))
            .fill_null(False)
            .alias(out_col)
        )
    )


def flag_wide_spread(
    df_long: pl.DataFrame,
    option_type: str,
    *,
    rel_spread_col: str = "rel_spread",
    mid_col: str = "mid_price",
    min_mid: float = 0.01,
    threshold: float = 1.0,
    out_col: str = "wide_spread_violation",
) -> pl.DataFrame:
    """Wide spread flag: rel_spread > threshold, only when mid >= min_mid."""
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    df_sub = df_long.filter(pl.col("option_type") == option_type)

    return df_sub.with_columns(
        pl.when(pl.col(mid_col) >= min_mid)
        .then(pl.col(rel_spread_col) > threshold)
        .otherwise(False)
        .fill_null(False)
        .alias(out_col)
    )