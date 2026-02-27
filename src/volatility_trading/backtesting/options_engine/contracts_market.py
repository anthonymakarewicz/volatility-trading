"""Market-facing quote contracts for options-engine runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """Typed quote snapshot used by options-engine core lifecycle logic.

    This object intentionally captures the subset of fields required by
    entry/sizing/valuation paths so core modules do not depend on raw row keys.
    """

    option_type_label: str
    strike: float
    bid_price: float
    ask_price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    expiry_date: pd.Timestamp | None = None
    dte: int | None = None
    spot_price: float | None = None
    smoothed_iv: float | None = None
    yte: float | None = None
    open_interest: float | None = None
    volume: float | None = None

    @classmethod
    def from_series(cls, quote: pd.Series) -> QuoteSnapshot:
        """Build a typed snapshot from one chain row.

        Required fields:
            - ``option_type``
            - ``strike``

        Optional numeric fields (prices and Greeks) default to ``0.0`` when
        absent so historical tests and lightweight adapters can still pass
        sparse rows without coupling to full chain schemas.
        """
        option_type_raw = quote.get("option_type")
        if option_type_raw is None:
            raise KeyError("option_type")
        return cls(
            option_type_label=str(option_type_raw),
            strike=_required_float(quote.get("strike"), field="strike"),
            bid_price=_float_or_default(quote.get("bid_price"), default=0.0),
            ask_price=_float_or_default(quote.get("ask_price"), default=0.0),
            delta=_float_or_default(quote.get("delta"), default=0.0),
            gamma=_float_or_default(quote.get("gamma"), default=0.0),
            vega=_float_or_default(quote.get("vega"), default=0.0),
            theta=_float_or_default(quote.get("theta"), default=0.0),
            expiry_date=_optional_timestamp(quote.get("expiry_date")),
            dte=_optional_int(quote.get("dte")),
            spot_price=_optional_float(quote.get("spot_price")),
            smoothed_iv=_optional_float(quote.get("smoothed_iv")),
            yte=_optional_float(quote.get("yte")),
            open_interest=_optional_float(quote.get("open_interest")),
            volume=_optional_float(quote.get("volume")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to a plain mapping."""
        return {
            "option_type": self.option_type_label,
            "strike": self.strike,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "expiry_date": self.expiry_date,
            "dte": self.dte,
            "spot_price": self.spot_price,
            "smoothed_iv": self.smoothed_iv,
            "yte": self.yte,
            "open_interest": self.open_interest,
            "volume": self.volume,
        }


def _required_float(value: object, *, field: str) -> float:
    if value is None or pd.isna(value):
        raise KeyError(field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _float_or_default(value: object, *, default: float) -> float:
    if value is None or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
