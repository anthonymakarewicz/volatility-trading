"""Row-level SOFT QC checks for ORATS option data."""

from __future__ import annotations

import polars as pl


def flag_theta_positive(
    df: pl.DataFrame,
    option_type: str,
    *,
    theta_col: str = "theta",
    eps: float = 1e-8,
    out_col: str = "theta_positive_violation",
) -> pl.DataFrame:
    """Flag positive theta beyond tolerance (diagnostic).

    Positive theta can happen legitimately (dividends/carry), so this is SOFT.
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    return df.filter(pl.col("option_type") == option_type).with_columns(
        (pl.col(theta_col) > eps).fill_null(False).alias(out_col)
    )


def flag_delta_bounds(
    df: pl.DataFrame,
    option_type: str,
    *,
    delta_col: str = "delta",
    eps: float = 1e-5,
    out_col: str = "delta_bounds_violation",
) -> pl.DataFrame:
    """Flag delta outside theoretical bounds (diagnostic).

    Calls: delta in [0, 1]
    Puts:  delta in [-1, 0]
    """
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'.")

    d = pl.col(delta_col)

    if option_type == "C":
        violation = (d < (0.0 - eps)) | (d > (1.0 + eps))
    else:  # "P"
        violation = (d < (-1.0 - eps)) | (d > (0.0 + eps))

    return df.filter(pl.col("option_type") == option_type).with_columns(
        violation.fill_null(False).alias(out_col)
    )


# -----------------------------------------------------------------------------
# Implied volatility checks
# -----------------------------------------------------------------------------


def flag_iv_high(
    df: pl.DataFrame,
    *,
    iv_col: str = "model_iv",
    threshold: float = 1.0,
    out_col: str = "iv_too_high_violation",
) -> pl.DataFrame:
    """Flag rows where IV is above a threshold (null-safe)."""
    return df.with_columns(
        (pl.col(iv_col).is_not_null() & (pl.col(iv_col) > threshold))
        .fill_null(False)
        .alias(out_col)
    )
