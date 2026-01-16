from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DownloadStrategy(Enum):
    FULL_HISTORY = "full_history"
    BY_TRADE_DATE = "by_trade_date"


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    strategy: DownloadStrategy
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


ENDPOINTS: dict[str, EndpointSpec] = {
    "monies_implied": EndpointSpec(
        path="/datav2/hist/monies/implied",
        required=("ticker", "tradeDate"),
        optional=("fields",),
        strategy=DownloadStrategy.BY_TRADE_DATE,
    ),
    "cores": EndpointSpec(
        path="/datav2/hist/cores",
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "summaries": EndpointSpec(
        path="/datav2/hist/summaries",
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "earnings": EndpointSpec(
        path="/datav2/hist/earnings",
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "dailies": EndpointSpec(
        path="/datav2/hist/dailies", # Daily OHLCV of underlying
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "hvs": EndpointSpec(
        path="/datav2/hist/hvs", # Historical volatilities
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "splits": EndpointSpec(
        path="/datav2/hist/splits", # Historical splits
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "ivrank": EndpointSpec(
        path="/datav2/hist/ivrank", # IV rank/percentile
        required=("ticker",),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
}


def get_endpoint_spec(endpoint: str) -> EndpointSpec:
    """Return the spec (path + required params) for a supported ORATS endpoint name."""
    try:
        return ENDPOINTS[endpoint]
    except KeyError as e:
        supported = ", ".join(sorted(ENDPOINTS.keys()))
        raise KeyError(
            f"Unknown ORATS endpoint '{endpoint}'. Supported: {supported}"
        ) from e