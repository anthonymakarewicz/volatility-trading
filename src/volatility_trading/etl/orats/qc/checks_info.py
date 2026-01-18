from __future__ import annotations

import polars as pl


def summarize_volume_oi_metrics(df: pl.DataFrame) -> dict[str, float | int]:
    """Informational volume/oi diagnostics."""
    n = df.height
    if n == 0:
        return {"n_rows": 0}

    metrics = df.select(
        pl.len().alias("n_rows"),
        (pl.col("volume") == 0).sum().alias("n_zero_volume"),
        (pl.col("open_interest") == 0).sum().alias("n_zero_open_interest"),
    ).to_dicts()[0]

    return {
        **metrics,
        "pct_zero_volume": metrics["n_zero_volume"] / n,
        "pct_zero_open_interest": metrics["n_zero_open_interest"] / n,
    }

def summarize_risk_free_rate_metrics(
    *,
    df: pl.DataFrame,
    rf_col: str = "risk_free_rate",
) -> dict[str, int]:
    """Informational risk-free rate diagnostics."""
    n = df.height
    if n == 0:
        return {"n_rows": 0}

    out = df.select(
        (pl.col(rf_col) < 0).sum().alias("n_rf_negative"),
        (pl.col(rf_col) > 0.10).sum().alias("n_rf_gt_10pct"),
        (pl.col(rf_col) > 1.00).sum().alias("n_rf_gt_100pct"),
    ).to_dicts()[0]

    # cast to int for JSON cleanliness
    return {k: int(v) for k, v in out.items()}