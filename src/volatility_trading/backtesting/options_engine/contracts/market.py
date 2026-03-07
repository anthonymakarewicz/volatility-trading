"""Market-facing contracts for options-engine runtime components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ...data_contracts import HedgeMarketData


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
    market_iv: float | None = None
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
            market_iv=_optional_float(quote.get("market_iv")),
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
            "market_iv": self.market_iv,
            "yte": self.yte,
            "open_interest": self.open_interest,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class HedgeMarketSnapshot:
    """Point-in-time hedge market snapshot used by hedging lifecycle logic."""

    mid: float
    bid: float
    ask: float
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.contract_multiplier) or self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be finite and > 0")

    @classmethod
    def missing(cls) -> HedgeMarketSnapshot:
        """Return a snapshot with unavailable prices and default scaling."""
        return cls(
            mid=float("nan"),
            bid=float("nan"),
            ask=float("nan"),
            contract_multiplier=1.0,
        )

    @classmethod
    def from_market_data(
        cls,
        *,
        hedge_market: HedgeMarketData | None,
        curr_date: pd.Timestamp,
    ) -> HedgeMarketSnapshot:
        """Resolve one-date hedge snapshot from user-provided market data."""
        if hedge_market is None:
            return cls.missing()
        return cls(
            mid=cls._resolve_series_price(series=hedge_market.mid, curr_date=curr_date),
            bid=cls._resolve_series_price(series=hedge_market.bid, curr_date=curr_date),
            ask=cls._resolve_series_price(series=hedge_market.ask, curr_date=curr_date),
            contract_multiplier=float(hedge_market.contract_multiplier),
        )

    @staticmethod
    def _resolve_series_price(
        *, series: pd.Series | None, curr_date: pd.Timestamp
    ) -> float:
        """Resolve one hedge series value at one date, defaulting to NaN."""
        if series is None:
            return float("nan")
        try:
            raw = series.loc[pd.Timestamp(curr_date)]
        except KeyError:
            return float("nan")
        if isinstance(raw, pd.Series):
            raw = raw.iloc[-1]
        if pd.isna(raw):
            return float("nan")
        return float(raw)


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
