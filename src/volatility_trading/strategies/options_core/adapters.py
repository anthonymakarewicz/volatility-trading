"""Adapters between strategy quote rows and option pricing/risk contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from volatility_trading.options import OptionLeg, OptionSpec, OptionType, PositionSide
from volatility_trading.options.types import OptionTypeInput

# TODO: Here we pass the lot_size (aka contratc multiplier around) but why not part it part of OptionLeg ?
# namely not a backtest parameter but tied to the dataclass itself (same with lot size for the hedgign instrument)


def normalize_chain_option_type(option_type: OptionTypeInput) -> OptionType:
    """Normalize vendor option side labels to canonical OptionType enum."""
    if isinstance(option_type, OptionType):
        return option_type
    if option_type in ("call", "C"):
        return OptionType.CALL
    if option_type in ("put", "P"):
        return OptionType.PUT
    raise ValueError(f"unsupported option type: {option_type!r}")


def option_type_to_chain_label(option_type: OptionTypeInput) -> str:
    """Map canonical option type to chain label used in ORATS panels."""
    canonical = normalize_chain_option_type(option_type)
    return "C" if canonical is OptionType.CALL else "P"


def time_to_expiry_years(
    *,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    quote_yte: float | int | None,
    quote_dte: float | int | None,
) -> float:
    """Return annualized maturity using quote fields with calendar fallback.

    Priority order:
    1. ``quote_yte`` when present and valid,
    2. ``quote_dte / 365``,
    3. calendar days between entry and expiry.
    """

    def _positive_or_none(value: float | int | None) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0:
            return None
        return numeric

    yte = _positive_or_none(quote_yte)
    if yte is not None:
        return yte

    dte = _positive_or_none(quote_dte)
    if dte is not None:
        return max(dte / 365.0, 1e-8)

    days = (pd.Timestamp(expiry_date) - pd.Timestamp(entry_date)).days
    return max(days / 365.0, 1e-8)


def quote_to_option_spec(
    *,
    quote: pd.Series,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
) -> OptionSpec:
    """Build ``OptionSpec`` from one selected chain row."""
    return OptionSpec(
        strike=float(quote["strike"]),
        time_to_expiry=time_to_expiry_years(
            entry_date=entry_date,
            expiry_date=expiry_date,
            quote_yte=quote.get("yte"),
            quote_dte=quote.get("dte"),
        ),
        option_type=normalize_chain_option_type(quote["option_type"]),
    )


def quote_to_option_leg(
    *,
    quote: pd.Series,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    entry_price: float,
    side: int,
    contract_multiplier: float,
) -> OptionLeg:
    """Build ``OptionLeg`` from selected quote row and execution metadata."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    return OptionLeg(
        spec=quote_to_option_spec(
            quote=quote,
            entry_date=entry_date,
            expiry_date=expiry_date,
        ),
        entry_price=float(entry_price),
        side=PositionSide(side),
        contract_multiplier=float(contract_multiplier),
    )
