"""Rate input models shared by backtesting margin and performance code.

The same rate abstraction is used for:
- financing carry in margin lifecycle simulation
- excess-return metrics (e.g., Sharpe) in performance reporting

This keeps rate handling consistent whether inputs are constants, date-indexed
series, or custom model objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol, TypeAlias, runtime_checkable

import pandas as pd


@runtime_checkable
class RateModel(Protocol):
    """Annualized rate provider evaluated at a date."""

    def annual_rate(self, as_of: pd.Timestamp | None = None) -> float:
        """Return annualized rate (decimal) for the provided date."""


RateInput: TypeAlias = float | int | pd.Series | RateModel


@dataclass(frozen=True)
class ConstantRateModel:
    """Constant annualized rate model."""

    rate_annual: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_annual < 0:
            raise ValueError("rate_annual must be >= 0")

    def annual_rate(self, as_of: pd.Timestamp | None = None) -> float:
        _ = as_of
        return float(self.rate_annual)


@dataclass(frozen=True)
class SeriesRateModel:
    """Date-indexed annualized rate model with as-of lookup.

    For any `as_of` date, the model returns the latest known rate on or before
    that date (forward-filled in time). If `as_of` is earlier than the first
    observation, the first observation is used.
    """

    series: pd.Series
    _prepared: pd.Series = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cleaned = pd.Series(self.series).dropna()
        if cleaned.empty:
            raise ValueError("series rate input must contain at least one non-null row")

        try:
            idx = pd.to_datetime(cleaned.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("series rate input must be indexed by dates") from exc

        prepared = pd.Series(cleaned.astype(float).values, index=idx).sort_index()
        if prepared.index.has_duplicates:
            prepared = prepared.groupby(level=0).last()
        if (prepared < 0).any():
            raise ValueError("series rates must be >= 0")

        object.__setattr__(self, "_prepared", prepared)

    def annual_rate(self, as_of: pd.Timestamp | None = None) -> float:
        if as_of is None:
            return float(self._prepared.iloc[-1])

        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts <= self._prepared.index[0]:
            return float(self._prepared.iloc[0])

        position = self._prepared.index.searchsorted(as_of_ts, side="right") - 1
        return float(self._prepared.iloc[int(position)])


def coerce_rate_model(value: RateInput) -> RateModel:
    """Normalize constant/series/model input into a `RateModel`."""
    if isinstance(value, RateModel):
        return value
    if isinstance(value, pd.Series):
        return SeriesRateModel(value)
    if isinstance(value, Real):
        return ConstantRateModel(float(value))
    raise TypeError(
        "rate input must be a numeric constant, pandas Series, or RateModel"
    )
