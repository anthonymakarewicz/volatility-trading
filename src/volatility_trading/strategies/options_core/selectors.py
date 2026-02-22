"""Shared quote/expiry selection helpers for option structure builders."""

from __future__ import annotations

import pandas as pd


def pick_quote_by_delta(
    df_leg: pd.DataFrame,
    *,
    target_delta: float,
    delta_tolerance: float = 0.05,
) -> pd.Series | None:
    """Pick the quote with closest delta to target inside tolerance."""
    df2 = df_leg.copy()
    df2["d_err"] = (df2["delta"] - target_delta).abs()
    df2 = df2[df2["d_err"] <= delta_tolerance]
    if df2.empty:
        return None
    return df2.iloc[df2["d_err"].values.argmin()]


def choose_expiry_by_target_dte(
    chain: pd.DataFrame,
    *,
    target_dte: int = 30,
    max_dte_diff: int = 7,
    min_atm_quotes: int = 2,
    dte_column: str = "dte",
    strike_column: str = "strike",
    spot_column: str = "spot_price",
) -> int | None:
    """Pick expiry nearest target DTE in-band with minimum ATM quote depth."""
    dtes = chain[dte_column].dropna().unique()
    candidates = [d for d in dtes if abs(d - target_dte) <= max_dte_diff]
    if not candidates:
        return None

    def _is_viable(dte: int) -> bool:
        sub = chain[chain[dte_column] == dte]
        spot = sub[spot_column].iloc[0]
        atm_band = (sub[strike_column] / spot).between(0.98, 1.02)
        atm_quotes = sub[atm_band]
        return len(atm_quotes) >= min_atm_quotes

    viable = [d for d in candidates if _is_viable(d)]
    if not viable:
        return None
    return min(viable, key=lambda d: abs(d - target_dte))
