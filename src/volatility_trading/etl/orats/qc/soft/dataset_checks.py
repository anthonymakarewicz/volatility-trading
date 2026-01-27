# qc/soft/checks_calendar.py
from __future__ import annotations

from typing import Any

import polars as pl

import exchange_calendars as xcals  # pip: exchange-calendars


# TODO: Refactor XNYS calendar functions into a shared utils module



def check_missing_sessions_xnys(
    *,
    df: pl.DataFrame,
    date_col: str = "trade_date",
    cal_name: str = "XNYS",
) -> dict[str, Any]:
    """
    Missing trading sessions between min(trade_date) and max(trade_date),
    based on XNYS calendar.

    Returns:
      - n_units: expected_sessions
      - n_viol: missing_sessions
      - viol_rate: missing_sessions / expected_sessions
      - missing_dates: list[str] (ISO)
    """
    if df.height == 0:
        return {"n_units": 0, "n_viol": 0, "viol_rate": 0.0, "missing_dates": []}

    # observed dates in dataset
    obs = (
        df.select(pl.col(date_col).cast(pl.Date).unique().sort())
        .get_column(date_col)
        .to_list()
    )
    if not obs:
        return {"n_units": 0, "n_viol": 0, "viol_rate": 0.0, "missing_dates": []}

    start = min(obs)
    end = max(obs)

    cal = xcals.get_calendar(cal_name)

    # Expected trading sessions in [start, end]
    sessions = cal.sessions_in_range(start, end)  # DatetimeIndex (UTC)
    expected = {d.date() for d in sessions.to_pydatetime()}

    observed = set(obs)

    missing = sorted(expected - observed)
    n_units = len(expected)
    n_viol = len(missing)
    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    return {
        "n_units": n_units,
        "n_viol": n_viol,
        "viol_rate": viol_rate,
        "start": str(start),
        "end": str(end),
        "missing_dates": [str(d) for d in missing],
    }


def check_non_trading_dates_present_xnys(
    *,
    df: pl.DataFrame,
    date_col: str = "trade_date",
    cal_name: str = "XNYS",
) -> dict[str, Any]:
    """
    Dates present in dataset that are NOT trading sessions per XNYS calendar
    (e.g., weekends/holidays).

    Returns:
      - n_units: observed_dates
      - n_viol: extra_dates
      - viol_rate: extra_dates / observed_dates
      - extra_dates: list[str] (ISO)
    """
    if df.height == 0:
        return {"n_units": 0, "n_viol": 0, "viol_rate": 0.0, "extra_dates": []}

    obs = (
        df.select(pl.col(date_col).cast(pl.Date).unique().sort())
        .get_column(date_col)
        .to_list()
    )
    if not obs:
        return {"n_units": 0, "n_viol": 0, "viol_rate": 0.0, "extra_dates": []}

    start = min(obs)
    end = max(obs)

    cal = xcals.get_calendar(cal_name)
    sessions = cal.sessions_in_range(start, end)
    expected = {d.date() for d in sessions.to_pydatetime()}

    observed = set(obs)

    extra = sorted(observed - expected)
    n_units = len(observed)
    n_viol = len(extra)
    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    return {
        "n_units": n_units,
        "n_viol": n_viol,
        "viol_rate": viol_rate,
        "start": str(start),
        "end": str(end),
        "extra_dates": [str(d) for d in extra],
    }


def check_unique_rf_rate_per_day_expiry(
    *,
    df: pl.DataFrame,
    trade_col: str = "trade_date",
    expiry_col: str = "expiry_date",
    r_col: str = "risk_free_rate",
    tol_abs: float = 1e-4,          # ~1bp absolute tolerance
    tol_rel: float = 0.0,           # optional relative tolerance
    rel_floor: float = 1e-6,        # avoid div by ~0
    max_examples: int = 5,
) -> dict[str, Any]:
    """
    Unique risk-free rate per (trade_date, expiry_date) sanity check.

    Returns (required keys):
      - viol_rate: float  (#viol_units / #units)
      - n_viol: int       (#(trade_date, expiry) groups violating)
      - n_units: int      (#(trade_date, expiry) groups examined)
    """
    required_cols = {trade_col, expiry_col, r_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # Only evaluate where r is present
    dfx = (
        df
        .select([trade_col, expiry_col, r_col])
        .filter(pl.col(r_col).is_not_null())
    )
    if dfx.height == 0:
        return {
            "viol_rate": 0.0,
            "n_viol": 0,
            "n_units": 0,
            "reason": "no non-null risk_free_rate rows",
        }

    # Group by (trade_date, expiry_date) and measure spread
    g = [trade_col, expiry_col]
    grp = (
        dfx.group_by(g)
        .agg(
            pl.col(r_col).min().alias("r_min"),
            pl.col(r_col).max().alias("r_max"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("r_max") - pl.col("r_min")).alias("r_spread"),
        )
    )

    # Tolerance: abs OR (rel * max(|rate|, floor))
    abs_bad = pl.col("r_spread") > tol_abs
    rel_bad = (
        pl.col("r_spread")
        > (tol_rel * pl.max_horizontal([pl.col("r_max").abs(), pl.lit(rel_floor)]))
    )
    viol_expr = abs_bad | rel_bad

    grp2 = grp.with_columns(viol_expr.alias("viol"))

    n_units = int(grp2.height)
    n_viol = int(grp2.select(pl.col("viol").sum().alias("n"))["n"][0])

    viol_rate = (n_viol / n_units) if n_units > 0 else 0.0

    # Some helpful debug payload: worst offenders
    examples_df = (
        grp2.filter(pl.col("viol"))
        .sort("r_spread", descending=True)
        .head(max_examples)
    )
    # Make dates JSON-safe
    for col, dtype in examples_df.schema.items():
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            examples_df = examples_df.with_columns(pl.col(col).cast(pl.Utf8))

    max_spread = float(grp2.select(pl.col("r_spread").max().alias("m"))["m"][0])

    return {
        "viol_rate": float(viol_rate),
        "n_viol": n_viol,
        "n_units": n_units,
        "max_spread": max_spread,
        "examples": examples_df.to_dicts(),
    }