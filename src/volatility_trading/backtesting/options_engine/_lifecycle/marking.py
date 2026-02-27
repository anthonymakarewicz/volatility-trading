"""One-date mark step context and snapshot builders."""

from __future__ import annotations

import pandas as pd

from volatility_trading.options import MarginModel, PriceModel

from ...config import BacktestRunConfig
from ..economics import roundtrip_commission_per_structure_contract
from ..records import MtmRecord
from ..state import OpenPosition
from .margining import (
    evaluate_mark_margin,
    maybe_refresh_margin_per_contract,
)
from .record_builders import build_mark_record
from .runtime_state import (
    LifecycleStepContext,
    MarkMarginSnapshot,
    MarkValuationSnapshot,
)
from .valuation import resolve_mark_valuation


def build_mark_step_context(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    cfg: BacktestRunConfig,
    equity_running: float,
) -> LifecycleStepContext:
    """Build shared one-date execution context for mark transitions."""
    return LifecycleStepContext(
        curr_date=curr_date,
        cfg=cfg,
        equity_running=equity_running,
        lot_size=cfg.execution.lot_size,
        roundtrip_commission_per_contract=roundtrip_commission_per_structure_contract(
            commission_per_leg=cfg.execution.commission_per_leg,
            legs=position.intent.legs,
        ),
    )


def build_mark_step_snapshots(
    *,
    position: OpenPosition,
    step: LifecycleStepContext,
    options: pd.DataFrame,
    margin_model: MarginModel | None,
    pricer: PriceModel,
) -> tuple[MarkValuationSnapshot, MarkMarginSnapshot, MtmRecord]:
    """Build valuation, margin, and base MTM snapshots for one mark date."""
    valuation = resolve_mark_valuation(
        position=position,
        curr_date=step.curr_date,
        options=options,
        lot_size=step.lot_size,
    )
    maybe_refresh_margin_per_contract(
        position=position,
        curr_date=step.curr_date,
        lot_size=step.lot_size,
        valuation=valuation,
        margin_model=margin_model,
        pricer=pricer,
    )
    margin = evaluate_mark_margin(
        position=position,
        curr_date=step.curr_date,
        equity_running=step.equity_running,
        valuation=valuation,
    )
    mtm_record = build_mark_record(
        position=position,
        curr_date=step.curr_date,
        valuation=valuation,
        margin=margin,
    )
    return valuation, margin, mtm_record
