"""Dataset-level SOFT QC checks for ORATS option data."""

from __future__ import annotations

from typing import Any

import exchange_calendars as xcals
import polars as pl


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
