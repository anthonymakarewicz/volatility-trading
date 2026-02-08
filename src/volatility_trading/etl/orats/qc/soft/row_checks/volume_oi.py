from __future__ import annotations

import polars as pl


def flag_zero_volume(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Flag rows where volume == 0."""
    df_sub = df.filter(pl.col("option_type") == option_type).with_columns(
        zero_volume_violation=(pl.col(volume_col) == 0).fill_null(False)
    )
    return df_sub


def flag_zero_open_interest(
    df: pl.DataFrame,
    option_type: str,
    *,
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where open interest == 0."""
    df_sub = df.filter(pl.col("option_type") == option_type).with_columns(
        zero_open_interest_violation=(pl.col(oi_col) == 0).fill_null(False)
    )
    return df_sub


def flag_zero_vol_pos_oi(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where volume == 0 but OI > 0."""
    df_sub = df.filter(pl.col("option_type") == option_type).with_columns(
        zero_vol_pos_oi_violation=(
            (pl.col(volume_col) == 0) & (pl.col(oi_col) > 0)
        ).fill_null(False)
    )
    return df_sub


def flag_pos_vol_zero_oi(
    df: pl.DataFrame,
    option_type: str,
    *,
    volume_col: str = "volume",
    oi_col: str = "open_interest",
) -> pl.DataFrame:
    """Flag rows where volume > 0 but OI == 0."""
    df_sub = df.filter(pl.col("option_type") == option_type).with_columns(
        pos_vol_zero_oi_violation=(
            (pl.col(volume_col) > 0) & (pl.col(oi_col) == 0)
        ).fill_null(False)
    )
    return df_sub
