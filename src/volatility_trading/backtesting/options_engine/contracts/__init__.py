"""Internal runtime contracts for options-engine execution boundaries.

These re-exports exist mainly for package-level ergonomics. Most users should
prefer ``volatility_trading.backtesting`` or
``volatility_trading.backtesting.options_engine`` imports.
"""

from .execution import (
    SinglePositionBuildOutputsFn,
    SinglePositionCanReenterFn,
    SinglePositionExecutionPlan,
    SinglePositionHooks,
    SinglePositionMarkFn,
    SinglePositionOpenFn,
    SinglePositionPrepareEntryFn,
)
from .market import HedgeMarketSnapshot, QuoteSnapshot
from .records import MtmMargin, MtmRecord, TradeLegRecord, TradeRecord
from .runtime import HedgeState, LifecycleStepResult, OpenPosition, PositionEntrySetup
from .structures import EntryIntent, FillPolicy, LegSelection, LegSpec, StructureSpec

__all__ = [
    "FillPolicy",
    "QuoteSnapshot",
    "HedgeMarketSnapshot",
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
