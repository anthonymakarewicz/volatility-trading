"""Public contracts package for options-engine runtime boundaries."""

from .execution import (
    SinglePositionBuildOutputsFn,
    SinglePositionCanReenterFn,
    SinglePositionExecutionPlan,
    SinglePositionHooks,
    SinglePositionMarkFn,
    SinglePositionOpenFn,
    SinglePositionPrepareEntryFn,
)
from .market import QuoteSnapshot
from .records import MtmMargin, MtmRecord, TradeLegRecord, TradeRecord
from .runtime import HedgeState, LifecycleStepResult, OpenPosition, PositionEntrySetup
from .structures import EntryIntent, FillPolicy, LegSelection, LegSpec, StructureSpec

__all__ = [
    "FillPolicy",
    "QuoteSnapshot",
    "LegSpec",
    "StructureSpec",
    "LegSelection",
    "EntryIntent",
    "MtmMargin",
    "MtmRecord",
    "TradeLegRecord",
    "TradeRecord",
    "PositionEntrySetup",
    "OpenPosition",
    "HedgeState",
    "LifecycleStepResult",
    "SinglePositionMarkFn",
    "SinglePositionPrepareEntryFn",
    "SinglePositionOpenFn",
    "SinglePositionCanReenterFn",
    "SinglePositionBuildOutputsFn",
    "SinglePositionHooks",
    "SinglePositionExecutionPlan",
]
