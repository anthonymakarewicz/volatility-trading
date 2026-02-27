"""Pure valuation helpers for options lifecycle execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from volatility_trading.backtesting.types import BacktestRunConfig
from volatility_trading.options.types import Greeks, MarketState

from ..adapters import option_type_to_chain_label
from ..economics import effective_leg_side, leg_units
from ..entry import chain_for_date
from ..records import TradeLegRecord
from ..state import OpenPosition
from ..types import EntryIntent, LegSelection, QuoteSnapshot
from .runtime_state import MarkValuationSnapshot


def exit_leg_price(quote: QuoteSnapshot, *, side: int, cfg: BacktestRunConfig) -> float:
    """Return executable exit price for one leg given effective side."""
    if side == -1:
        return float(quote.ask_price + cfg.execution.slip_ask)
    if side == 1:
        return float(quote.bid_price - cfg.execution.slip_bid)
    raise ValueError("side must be -1 or +1")


def match_leg_quote(
    *,
    chain: pd.DataFrame,
    leg: LegSelection,
) -> QuoteSnapshot | None:
    """Return the row matching one leg strike/type for ``chain`` date slice."""
    strike = leg.quote.strike
    canonical_label = option_type_to_chain_label(leg.spec.option_type)
    candidates = chain
    if "expiry_date" in candidates.columns and leg.quote.expiry_date is not None:
        leg_expiry = pd.Timestamp(leg.quote.expiry_date)
        candidates = candidates[pd.to_datetime(candidates["expiry_date"]) == leg_expiry]
    candidates = candidates[
        (candidates["option_type"] == canonical_label)
        & (candidates["strike"] == strike)
    ]
    if candidates.empty:
        # Fallback for datasets using vendor labels already present in the leg row.
        vendor_label = leg.quote.option_type_label
        candidates = chain
        if "expiry_date" in candidates.columns and leg.quote.expiry_date is not None:
            leg_expiry = pd.Timestamp(leg.quote.expiry_date)
            candidates = candidates[
                pd.to_datetime(candidates["expiry_date"]) == leg_expiry
            ]
        candidates = candidates[
            (candidates["option_type"] == vendor_label)
            & (candidates["strike"] == strike)
        ]
    if candidates.empty:
        return None
    return QuoteSnapshot.from_series(candidates.iloc[0])


def greeks_per_contract(
    *,
    leg_quotes: Sequence[tuple[LegSelection, QuoteSnapshot]],
    lot_size: int,
) -> Greeks:
    """Aggregate structure Greeks for one strategy contract unit."""
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    for leg, quote in leg_quotes:
        side = effective_leg_side(leg)
        units = leg_units(leg)
        delta += side * float(quote.delta) * units * lot_size
        gamma += side * float(quote.gamma) * units * lot_size
        vega += side * float(quote.vega) * units * lot_size
        theta += side * float(quote.theta) * units * lot_size
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta)


def mark_to_mid(
    *,
    legs: Sequence[LegSelection],
    leg_quotes: Sequence[QuoteSnapshot],
    lot_size: int,
    contracts_open: int,
    net_entry: float,
) -> float:
    """Return cumulative MTM PnL from current mids versus entry notional."""
    current_value = 0.0
    for leg, quote in zip(legs, leg_quotes, strict=True):
        mid = 0.5 * (float(quote.bid_price) + float(quote.ask_price))
        current_value += effective_leg_side(leg) * mid * leg_units(leg)
    return current_value * lot_size * contracts_open - net_entry


def pnl_per_contract_from_exit_prices(
    *,
    legs: Sequence[LegSelection],
    exit_prices: Sequence[float],
    lot_size: int,
) -> float:
    """Return realized PnL for one strategy contract at given exit prices."""
    pnl_pc = 0.0
    for leg, exit_price in zip(legs, exit_prices, strict=True):
        pnl_pc += (
            effective_leg_side(leg)
            * (float(exit_price) - float(leg.entry_price))
            * leg_units(leg)
            * lot_size
        )
    return pnl_pc


def entry_net_notional(
    *,
    legs: Sequence[LegSelection],
    lot_size: int,
    contracts: int,
) -> float:
    """Return signed entry premium of the full opened structure."""
    return sum(
        effective_leg_side(leg)
        * float(leg.entry_price)
        * leg_units(leg)
        * lot_size
        * contracts
        for leg in legs
    )


def summary_expiry_and_dte_from_legs(
    *,
    entry_date: pd.Timestamp,
    legs: Sequence[LegSelection],
) -> tuple[pd.Timestamp | None, int | None]:
    """Return single-expiry summary fields when all legs share them."""
    expiries: list[pd.Timestamp] = []
    dtes: list[int] = []
    for leg in legs:
        if leg.quote.expiry_date is not None:
            expiries.append(pd.Timestamp(leg.quote.expiry_date))
        if leg.quote.dte is not None:
            dtes.append(int(leg.quote.dte))

    expiry_summary: pd.Timestamp | None = None
    if expiries and all(exp == expiries[0] for exp in expiries):
        expiry_summary = expiries[0]

    dte_summary: int | None = None
    if dtes and all(dte == dtes[0] for dte in dtes):
        dte_summary = dtes[0]
    elif expiry_summary is not None:
        dte_summary = max((expiry_summary - pd.Timestamp(entry_date)).days, 1)

    return expiry_summary, dte_summary


def trade_legs_payload(
    *,
    legs: Sequence[LegSelection],
    exit_prices: Sequence[float] | None = None,
) -> tuple[TradeLegRecord, ...]:
    """Build per-leg trade payload for durable multi-expiry trade records."""
    payload: list[TradeLegRecord] = []
    for idx, leg in enumerate(legs):
        expiry_date = (
            pd.Timestamp(leg.quote.expiry_date)
            if leg.quote.expiry_date is not None
            else None
        )
        strike = float(leg.quote.strike)
        exit_price = (
            float(exit_prices[idx])
            if exit_prices is not None and idx < len(exit_prices)
            else np.nan
        )
        payload.append(
            TradeLegRecord(
                leg_index=idx,
                option_type=str(leg.spec.option_type),
                strike=strike,
                expiry_date=expiry_date,
                weight=int(leg.spec.weight),
                side=int(leg.side),
                effective_side=effective_leg_side(leg),
                entry_price=float(leg.entry_price),
                exit_price=exit_price,
                delta_target=float(leg.spec.delta_target),
                delta_tolerance=float(leg.spec.delta_tolerance),
                expiry_group=leg.spec.expiry_group,
            )
        )
    return tuple(payload)


def intent_with_updated_quotes(
    *,
    intent: EntryIntent,
    leg_quotes: Sequence[QuoteSnapshot],
) -> EntryIntent:
    """Clone intent replacing quote snapshots with current-date rows."""
    legs = tuple(
        LegSelection(
            spec=leg.spec,
            quote=quote,
            side=leg.side,
            entry_price=leg.entry_price,
        )
        for leg, quote in zip(intent.legs, leg_quotes, strict=True)
    )
    return EntryIntent(
        entry_date=intent.entry_date,
        expiry_date=intent.expiry_date,
        chosen_dte=intent.chosen_dte,
        legs=legs,
        entry_state=intent.entry_state,
    )


def resolve_mark_valuation(
    *,
    position: OpenPosition,
    curr_date: pd.Timestamp,
    options: pd.DataFrame,
    lot_size: int,
) -> MarkValuationSnapshot:
    """Resolve one-date MTM and Greeks snapshot before margin/exit handling."""
    chain_all = chain_for_date(options, curr_date)
    leg_quotes = tuple(
        match_leg_quote(chain=chain_all, leg=leg) for leg in position.intent.legs
    )

    prev_mtm_before = position.prev_mtm
    has_missing_quote = any(quote is None for quote in leg_quotes)
    if has_missing_quote:
        pnl_mtm = prev_mtm_before
        greeks = position.last_greeks
        complete_leg_quotes: tuple[QuoteSnapshot, ...] | None = None
    else:
        complete_leg_quotes = tuple(quote for quote in leg_quotes if quote is not None)
        pnl_mtm = mark_to_mid(
            legs=position.intent.legs,
            leg_quotes=complete_leg_quotes,
            lot_size=lot_size,
            contracts_open=position.contracts_open,
            net_entry=position.net_entry,
        )
        greeks_pc = greeks_per_contract(
            leg_quotes=tuple(
                zip(position.intent.legs, complete_leg_quotes, strict=True)
            ),
            lot_size=lot_size,
        )
        greeks = greeks_pc.scaled(position.contracts_open)

    spot_curr = position.last_market.spot
    if "spot_price" in chain_all.columns and not chain_all.empty:
        spot_curr = float(chain_all["spot_price"].iloc[0])

    iv_curr = position.last_market.volatility
    if (
        complete_leg_quotes is not None
        and all(quote.smoothed_iv is not None for quote in complete_leg_quotes)
        and len(complete_leg_quotes) > 0
    ):
        iv_curr = float(
            np.mean([float(quote.smoothed_iv) for quote in complete_leg_quotes])
        )

    hedge_pnl = 0.0
    net_delta = position.last_net_delta if has_missing_quote else greeks.delta
    delta_pnl_market = (pnl_mtm - prev_mtm_before) + hedge_pnl
    market = MarketState(spot=spot_curr, volatility=iv_curr)
    return MarkValuationSnapshot(
        prev_mtm_before=prev_mtm_before,
        pnl_mtm=pnl_mtm,
        greeks=greeks,
        complete_leg_quotes=complete_leg_quotes,
        has_missing_quote=has_missing_quote,
        market=market,
        hedge_pnl=hedge_pnl,
        net_delta=net_delta,
        delta_pnl_market=delta_pnl_market,
    )


def update_position_mark_state(
    *,
    position: OpenPosition,
    pnl_mtm: float,
    market: MarketState,
    greeks: Greeks,
    net_delta: float,
) -> None:
    """Persist updated in-trade state after one mark step."""
    position.prev_mtm = pnl_mtm
    position.last_market = market
    position.last_greeks = greeks
    position.last_net_delta = net_delta


def exit_prices_for_position(
    *,
    position: OpenPosition,
    leg_quotes: tuple[QuoteSnapshot, ...],
    cfg: BacktestRunConfig,
) -> tuple[float, ...]:
    """Compute executable exit prices for all legs of one open position."""
    return tuple(
        exit_leg_price(
            quote,
            side=effective_leg_side(leg),
            cfg=cfg,
        )
        for leg, quote in zip(position.intent.legs, leg_quotes, strict=True)
    )
