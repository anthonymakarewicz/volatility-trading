"""One-date mark step context and snapshot builders."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from volatility_trading.options import MarginModel, PriceModel

from ...config import BacktestRunConfig
from ...data_contracts import HedgeMarketData
from ..contracts.records import MtmRecord
from ..contracts.runtime import OpenPosition
from ..economics import roundtrip_commission_per_structure_contract
from ..specs import DeltaHedgePolicy
from .hedging import (
    DeltaHedgeEngine,
    HedgeExecutionModel,
    HedgeTargetModel,
)
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
    delta_hedge_policy: DeltaHedgePolicy,
    hedge_market: HedgeMarketData | None,
    hedge_target_model: HedgeTargetModel,
    hedge_execution_model: HedgeExecutionModel,
) -> tuple[MarkValuationSnapshot, MarkMarginSnapshot, MtmRecord]:
    """Build valuation, margin, and base MTM snapshots for one mark date."""
    valuation = resolve_mark_valuation(
        position=position,
        curr_date=step.curr_date,
        options=options,
        lot_size=step.lot_size,
    )
    hedger = DeltaHedgeEngine(
        delta_hedge_policy,
        target_model=hedge_target_model,
        execution_model=hedge_execution_model,
    )
    hedge_price = _resolve_hedge_price(
        hedge_market=hedge_market,
        curr_date=step.curr_date,
    )
    hedge_step = hedger.apply(
        position=position,
        curr_date=step.curr_date,
        option_delta=float(valuation.greeks.delta),
        hedge_price=hedge_price,
        execution=step.cfg.execution,
    )
    valuation = replace(
        valuation,
        hedge_pnl=hedge_step.hedge_pnl,
        net_delta=hedge_step.net_delta,
        delta_pnl_market=(valuation.pnl_mtm - valuation.prev_mtm_before)
        + hedge_step.hedge_pnl,
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
        hedge_price_prev=hedge_step.hedge_price_prev,
    )
    return valuation, margin, mtm_record


def _resolve_hedge_price(
    *,
    hedge_market: HedgeMarketData | None,
    curr_date: pd.Timestamp,
) -> float:
    """Resolve hedge mid price for one date from typed hedge market data."""
    if hedge_market is None:
        return float("nan")
    try:
        raw = hedge_market.mid.loc[pd.Timestamp(curr_date)]
    except KeyError:
        return float("nan")
    if isinstance(raw, pd.Series):
        raw = raw.iloc[-1]
    if pd.isna(raw):
        return float("nan")
    return float(raw)
