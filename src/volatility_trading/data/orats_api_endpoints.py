from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DownloadStrategy(Enum):
    FULL_HISTORY = "full_history"
    BY_TRADE_DATE = "by_trade_date"


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    strategy: DownloadStrategy


ENDPOINTS: dict[str, EndpointSpec] = {
    "monies_implied": EndpointSpec(
        path="/datav2/hist/monies/implied",
        required=("ticker", "tradeDate"),
        optional=("fields",),
        strategy=DownloadStrategy.BY_TRADE_DATE,
    ),
    "cores": EndpointSpec(
        path="/datav2/hist/cores",
        required=("ticker", "tradeDate"),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
    "summaries": EndpointSpec(
        path="/datav2/hist/summaries",
        required=("ticker", "tradeDate"),
        optional=("fields",),
        strategy=DownloadStrategy.FULL_HISTORY,
    ),
}