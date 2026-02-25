"""Shared open/mark/close lifecycle engine for arbitrary option structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from volatility_trading.options import MarginModel, PriceModel

from ..margin import MarginAccount, MarginPolicy, MarginStatus
from ..types import BacktestConfig
from .adapters import option_type_to_chain_label
from .exit_rules import ExitRuleSet
from .sizing import estimate_entry_intent_margin_per_contract
from .types import EntryIntent, LegSelection


def _effective_leg_side(leg: LegSelection) -> int:
    """Return signed side after applying potential negative leg weights."""
    weight_sign = 1 if leg.spec.weight >= 0 else -1
    return int(leg.side) * weight_sign


def _leg_units(leg: LegSelection) -> int:
    """Return absolute leg ratio multiplier for PnL/Greek aggregation."""
    return abs(int(leg.spec.weight))


@dataclass(frozen=True)
class PositionEntrySetup:
    """Entry payload consumed by the generic lifecycle engine.

    Attributes:
        intent: Resolved structure entry intent.
        contracts: Number of structure contracts to open.
        risk_per_contract: Optional worst-case risk estimate used for sizing.
        risk_worst_scenario: Optional label for the worst risk scenario.
        margin_per_contract: Optional initial margin estimate per contract.
    """

    intent: EntryIntent
    contracts: int
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_per_contract: float | None


@dataclass
class OpenPosition:
    """Mutable open-position state updated once per trading date.

    This object stores lifecycle state required to mark, hedge, finance, and
    close a position consistently across dates.
    """

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp | None
    chosen_dte: int | None
    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None
    intent: EntryIntent
    contracts_open: int
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    net_entry: float
    prev_mtm: float
    hedge_qty: float
    hedge_price_entry: float
    last_spot: float
    last_iv: float
    last_delta: float
    last_gamma: float
    last_vega: float
    last_theta: float
    last_net_delta: float


@dataclass(frozen=True)
class _EntryMarginSnapshot:
    """Entry-day margin/accounting snapshot used to build records/state."""

    margin_account: MarginAccount | None
    latest_margin_per_contract: float | None
    initial_margin_requirement: float
    entry_delta_pnl: float
    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int


@dataclass(frozen=True)
class _MarkValuationSnapshot:
    """One-date valuation snapshot before margin and exits are applied."""

    prev_mtm_before: float
    pnl_mtm: float
    delta: float
    gamma: float
    vega: float
    theta: float
    complete_leg_quotes: tuple[pd.Series, ...] | None
    has_missing_quote: bool
    spot: float
    iv: float
    hedge_pnl: float
    net_delta: float
    delta_pnl_market: float


@dataclass(frozen=True)
class _MarkMarginSnapshot:
    """One-date margin/accounting state after evaluating the margin account."""

    initial_margin_requirement: float
    financing_pnl: float
    maintenance_margin_requirement: float
    margin_excess: float
    margin_deficit: float
    in_margin_call: bool
    margin_call_days: int
    forced_liquidation: bool
    contracts_liquidated: int
    margin_status: MarginStatus | None


@dataclass(frozen=True)
class PositionLifecycleEngine:
    """Shared position lifecycle logic: open, mark, margin-manage, and close."""

    rebalance_period: int | None
    max_holding_period: int | None
    exit_rule_set: ExitRuleSet
    margin_policy: MarginPolicy | None
    margin_model: MarginModel | None
    pricer: PriceModel

    @staticmethod
    def _chain_for_date(
        options: pd.DataFrame, trade_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Return a date slice as DataFrame even when a single row is returned."""
        chain = options.loc[trade_date]
        if isinstance(chain, pd.Series):
            return chain.to_frame().T
        return chain

    @staticmethod
    def _exit_leg_price(quote: pd.Series, *, side: int, cfg: BacktestConfig) -> float:
        """Return executable exit price for one leg given effective side."""
        if side == -1:
            return float(quote["ask_price"] + cfg.slip_ask)
        if side == 1:
            return float(quote["bid_price"] - cfg.slip_bid)
        raise ValueError("side must be -1 or +1")

    @staticmethod
    def _match_leg_quote(
        *,
        chain: pd.DataFrame,
        leg: LegSelection,
    ) -> pd.Series | None:
        """Return the row matching one leg strike/type for `chain` date slice."""
        strike = leg.quote["strike"]
        canonical_label = option_type_to_chain_label(leg.spec.option_type)
        candidates = chain
        if "expiry_date" in candidates.columns and "expiry_date" in leg.quote.index:
            leg_expiry = pd.Timestamp(leg.quote["expiry_date"])
            candidates = candidates[
                pd.to_datetime(candidates["expiry_date"]) == leg_expiry
            ]
        candidates = candidates[
            (candidates["option_type"] == canonical_label)
            & (candidates["strike"] == strike)
        ]
        if candidates.empty:
            # Fallback for datasets using vendor labels already present in the leg row.
            vendor_label = leg.quote.get("option_type")
            candidates = chain
            if "expiry_date" in candidates.columns and "expiry_date" in leg.quote.index:
                leg_expiry = pd.Timestamp(leg.quote["expiry_date"])
                candidates = candidates[
                    pd.to_datetime(candidates["expiry_date"]) == leg_expiry
                ]
            candidates = candidates[
                (candidates["option_type"] == vendor_label)
                & (candidates["strike"] == strike)
            ]
        if candidates.empty:
            return None
        return candidates.iloc[0]

    @staticmethod
    def _greeks_per_contract(
        *,
        leg_quotes: Sequence[tuple[LegSelection, pd.Series]],
        lot_size: int,
    ) -> tuple[float, float, float, float]:
        """Aggregate structure Greeks for one strategy contract unit."""
        delta = 0.0
        gamma = 0.0
        vega = 0.0
        theta = 0.0
        for leg, quote in leg_quotes:
            side = _effective_leg_side(leg)
            units = _leg_units(leg)
            delta += side * float(quote["delta"]) * units * lot_size
            gamma += side * float(quote["gamma"]) * units * lot_size
            vega += side * float(quote["vega"]) * units * lot_size
            theta += side * float(quote["theta"]) * units * lot_size
        return delta, gamma, vega, theta

    @staticmethod
    def _mark_to_mid(
        *,
        legs: Sequence[LegSelection],
        leg_quotes: Sequence[pd.Series],
        lot_size: int,
        contracts_open: int,
        net_entry: float,
    ) -> float:
        """Return cumulative MTM PnL from current mids versus entry notional."""
        current_value = 0.0
        for leg, quote in zip(legs, leg_quotes, strict=True):
            mid = 0.5 * (float(quote["bid_price"]) + float(quote["ask_price"]))
            current_value += _effective_leg_side(leg) * mid * _leg_units(leg)
        return current_value * lot_size * contracts_open - net_entry

    @staticmethod
    def _pnl_per_contract_from_exit_prices(
        *,
        legs: Sequence[LegSelection],
        exit_prices: Sequence[float],
        lot_size: int,
    ) -> float:
        """Return realized PnL for one strategy contract at given exit prices."""
        pnl_pc = 0.0
        for leg, exit_price in zip(legs, exit_prices, strict=True):
            pnl_pc += (
                _effective_leg_side(leg)
                * (float(exit_price) - float(leg.entry_price))
                * _leg_units(leg)
                * lot_size
            )
        return pnl_pc

    @staticmethod
    def _entry_net_notional(
        *,
        legs: Sequence[LegSelection],
        lot_size: int,
        contracts: int,
    ) -> float:
        """Return signed entry premium of the full opened structure."""
        return sum(
            _effective_leg_side(leg)
            * float(leg.entry_price)
            * _leg_units(leg)
            * lot_size
            * contracts
            for leg in legs
        )

    @staticmethod
    def _summary_expiry_and_dte_from_legs(
        *,
        entry_date: pd.Timestamp,
        legs: Sequence[LegSelection],
    ) -> tuple[pd.Timestamp | None, int | None]:
        """Return single-expiry summary fields when all legs share them."""
        expiries: list[pd.Timestamp] = []
        dtes: list[int] = []
        for leg in legs:
            if "expiry_date" in leg.quote.index and pd.notna(leg.quote["expiry_date"]):
                expiries.append(pd.Timestamp(leg.quote["expiry_date"]))
            if "dte" in leg.quote.index and pd.notna(leg.quote["dte"]):
                dtes.append(int(leg.quote["dte"]))

        expiry_summary: pd.Timestamp | None = None
        if expiries and all(exp == expiries[0] for exp in expiries):
            expiry_summary = expiries[0]

        dte_summary: int | None = None
        if dtes and all(dte == dtes[0] for dte in dtes):
            dte_summary = dtes[0]
        elif expiry_summary is not None:
            dte_summary = max((expiry_summary - pd.Timestamp(entry_date)).days, 1)

        return expiry_summary, dte_summary

    @staticmethod
    def _trade_legs_payload(
        *,
        legs: Sequence[LegSelection],
        exit_prices: Sequence[float] | None = None,
    ) -> list[dict]:
        """Build per-leg trade payload for durable multi-expiry trade records."""
        payload: list[dict] = []
        for idx, leg in enumerate(legs):
            expiry_date = (
                pd.Timestamp(leg.quote["expiry_date"])
                if "expiry_date" in leg.quote.index
                and pd.notna(leg.quote["expiry_date"])
                else None
            )
            strike = (
                float(leg.quote["strike"])
                if "strike" in leg.quote.index and pd.notna(leg.quote["strike"])
                else np.nan
            )
            exit_price = (
                float(exit_prices[idx])
                if exit_prices is not None and idx < len(exit_prices)
                else np.nan
            )
            payload.append(
                {
                    "leg_index": idx,
                    "option_type": str(leg.spec.option_type),
                    "strike": strike,
                    "expiry_date": expiry_date,
                    "weight": int(leg.spec.weight),
                    "side": int(leg.side),
                    "effective_side": _effective_leg_side(leg),
                    "entry_price": float(leg.entry_price),
                    "exit_price": exit_price,
                    "delta_target": float(leg.spec.delta_target),
                    "delta_tolerance": float(leg.spec.delta_tolerance),
                    "expiry_group": leg.spec.expiry_group,
                }
            )
        return payload

    @staticmethod
    def _intent_with_updated_quotes(
        *,
        intent: EntryIntent,
        leg_quotes: Sequence[pd.Series],
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

    def _evaluate_entry_margin(
        self,
        *,
        setup: PositionEntrySetup,
        equity_running: float,
        contracts_open: int,
        roundtrip_commission_per_contract: float,
    ) -> _EntryMarginSnapshot:
        """Evaluate entry-day margin and financing state for one opened position."""
        margin_account = (
            MarginAccount(self.margin_policy)
            if self.margin_policy is not None
            else None
        )
        latest_margin_per_contract = setup.margin_per_contract
        initial_margin_requirement = (
            latest_margin_per_contract or 0.0
        ) * contracts_open
        entry_delta_pnl = -(roundtrip_commission_per_contract * contracts_open)

        financing_pnl = 0.0
        maintenance_margin_requirement = 0.0
        margin_excess = np.nan
        margin_deficit = np.nan
        in_margin_call = False
        margin_call_days = 0
        forced_liquidation = False
        contracts_liquidated = 0

        if margin_account is not None:
            margin_status = margin_account.evaluate(
                equity=equity_running + entry_delta_pnl,
                initial_margin_requirement=initial_margin_requirement,
                open_contracts=contracts_open,
                as_of=setup.intent.entry_date,
            )
            financing_pnl = margin_status.financing_pnl
            entry_delta_pnl += financing_pnl
            maintenance_margin_requirement = (
                margin_status.maintenance_margin_requirement
            )
            margin_excess = margin_status.margin_excess
            margin_deficit = margin_status.margin_deficit
            in_margin_call = margin_status.in_margin_call
            margin_call_days = margin_status.margin_call_days
            forced_liquidation = margin_status.forced_liquidation
            contracts_liquidated = margin_status.contracts_to_liquidate

        return _EntryMarginSnapshot(
            margin_account=margin_account,
            latest_margin_per_contract=latest_margin_per_contract,
            initial_margin_requirement=initial_margin_requirement,
            entry_delta_pnl=entry_delta_pnl,
            financing_pnl=financing_pnl,
            maintenance_margin_requirement=maintenance_margin_requirement,
            margin_excess=margin_excess,
            margin_deficit=margin_deficit,
            in_margin_call=in_margin_call,
            margin_call_days=margin_call_days,
            forced_liquidation=forced_liquidation,
            contracts_liquidated=contracts_liquidated,
        )

    @staticmethod
    def _build_entry_record(
        *,
        setup: PositionEntrySetup,
        contracts_open: int,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        net_delta: float,
        margin: _EntryMarginSnapshot,
    ) -> dict:
        """Build entry-day MTM record from entry and margin snapshots."""
        return {
            "date": setup.intent.entry_date,
            "S": (
                setup.intent.entry_state.spot
                if setup.intent.entry_state is not None
                else np.nan
            ),
            "iv": (
                setup.intent.entry_state.volatility
                if setup.intent.entry_state is not None
                else np.nan
            ),
            "delta_pnl": margin.entry_delta_pnl,
            "delta": delta,
            "net_delta": net_delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "hedge_qty": 0.0,
            "hedge_price_prev": np.nan,
            "hedge_pnl": 0.0,
            "open_contracts": contracts_open,
            "margin_per_contract": margin.latest_margin_per_contract,
            "initial_margin_requirement": margin.initial_margin_requirement,
            "maintenance_margin_requirement": margin.maintenance_margin_requirement,
            "margin_excess": margin.margin_excess,
            "margin_deficit": margin.margin_deficit,
            "in_margin_call": margin.in_margin_call,
            "margin_call_days": margin.margin_call_days,
            "forced_liquidation": margin.forced_liquidation,
            "contracts_liquidated": margin.contracts_liquidated,
            "financing_pnl": margin.financing_pnl,
        }

    def _build_open_position(
        self,
        *,
        setup: PositionEntrySetup,
        contracts_open: int,
        net_entry: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        net_delta: float,
        margin: _EntryMarginSnapshot,
    ) -> OpenPosition:
        """Build mutable open-position state from entry lifecycle snapshots."""
        rebalance_date = (
            setup.intent.entry_date + pd.Timedelta(days=self.rebalance_period)
            if self.rebalance_period is not None
            else None
        )
        max_hold_date = (
            setup.intent.entry_date + pd.Timedelta(days=self.max_holding_period)
            if self.max_holding_period is not None
            else None
        )
        expiry_summary, dte_summary = self._summary_expiry_and_dte_from_legs(
            entry_date=setup.intent.entry_date,
            legs=setup.intent.legs,
        )

        return OpenPosition(
            entry_date=setup.intent.entry_date,
            expiry_date=expiry_summary,
            chosen_dte=dte_summary,
            rebalance_date=rebalance_date,
            max_hold_date=max_hold_date,
            intent=setup.intent,
            contracts_open=contracts_open,
            risk_per_contract=setup.risk_per_contract,
            risk_worst_scenario=setup.risk_worst_scenario,
            margin_account=margin.margin_account,
            latest_margin_per_contract=margin.latest_margin_per_contract,
            net_entry=net_entry,
            prev_mtm=0.0,
            hedge_qty=0.0,
            hedge_price_entry=np.nan,
            last_spot=(
                float(setup.intent.entry_state.spot)
                if setup.intent.entry_state is not None
                else float("nan")
            ),
            last_iv=(
                float(setup.intent.entry_state.volatility)
                if setup.intent.entry_state is not None
                else float("nan")
            ),
            last_delta=delta,
            last_gamma=gamma,
            last_vega=vega,
            last_theta=theta,
            last_net_delta=net_delta,
        )

    def _resolve_mark_valuation(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        lot_size: int,
    ) -> _MarkValuationSnapshot:
        """Resolve one-date MTM and Greeks snapshot before margin/exit handling."""
        chain_all = self._chain_for_date(options, curr_date)
        leg_quotes = tuple(
            self._match_leg_quote(chain=chain_all, leg=leg)
            for leg in position.intent.legs
        )

        prev_mtm_before = position.prev_mtm
        has_missing_quote = any(quote is None for quote in leg_quotes)
        if has_missing_quote:
            pnl_mtm = prev_mtm_before
            delta = position.last_delta
            gamma = position.last_gamma
            vega = position.last_vega
            theta = position.last_theta
            complete_leg_quotes: tuple[pd.Series, ...] | None = None
        else:
            complete_leg_quotes = tuple(
                quote for quote in leg_quotes if quote is not None
            )
            pnl_mtm = self._mark_to_mid(
                legs=position.intent.legs,
                leg_quotes=complete_leg_quotes,
                lot_size=lot_size,
                contracts_open=position.contracts_open,
                net_entry=position.net_entry,
            )
            delta_pc, gamma_pc, vega_pc, theta_pc = self._greeks_per_contract(
                leg_quotes=tuple(
                    zip(position.intent.legs, complete_leg_quotes, strict=True)
                ),
                lot_size=lot_size,
            )
            delta = delta_pc * position.contracts_open
            gamma = gamma_pc * position.contracts_open
            vega = vega_pc * position.contracts_open
            theta = theta_pc * position.contracts_open

        spot_curr = position.last_spot
        if "spot_price" in chain_all.columns and not chain_all.empty:
            spot_curr = float(chain_all["spot_price"].iloc[0])

        iv_curr = position.last_iv
        if (
            complete_leg_quotes is not None
            and all("smoothed_iv" in quote.index for quote in complete_leg_quotes)
            and len(complete_leg_quotes) > 0
        ):
            iv_curr = float(
                np.mean([float(quote["smoothed_iv"]) for quote in complete_leg_quotes])
            )

        hedge_pnl = 0.0
        net_delta = delta
        delta_pnl_market = (pnl_mtm - prev_mtm_before) + hedge_pnl
        return _MarkValuationSnapshot(
            prev_mtm_before=prev_mtm_before,
            pnl_mtm=pnl_mtm,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            complete_leg_quotes=complete_leg_quotes,
            has_missing_quote=has_missing_quote,
            spot=spot_curr,
            iv=iv_curr,
            hedge_pnl=hedge_pnl,
            net_delta=net_delta,
            delta_pnl_market=delta_pnl_market,
        )

    def _maybe_refresh_margin_per_contract(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        lot_size: int,
        valuation: _MarkValuationSnapshot,
    ) -> None:
        """Refresh per-contract margin estimate in-place when market state is valid."""
        if (
            self.margin_model is None
            or valuation.complete_leg_quotes is None
            or not np.isfinite(valuation.iv)
            or valuation.iv <= 0
        ):
            return

        current_intent = self._intent_with_updated_quotes(
            intent=position.intent,
            leg_quotes=valuation.complete_leg_quotes,
        )
        margin_pc_curr = estimate_entry_intent_margin_per_contract(
            intent=current_intent,
            as_of_date=curr_date,
            lot_size=lot_size,
            spot=float(valuation.spot),
            volatility=float(valuation.iv),
            margin_model=self.margin_model,
            pricer=self.pricer,
        )
        if margin_pc_curr is not None:
            position.latest_margin_per_contract = margin_pc_curr

    def _evaluate_mark_margin(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        equity_running: float,
        valuation: _MarkValuationSnapshot,
    ) -> _MarkMarginSnapshot:
        """Evaluate one-date margin snapshot after MTM and before exits."""
        initial_margin_requirement = (
            position.latest_margin_per_contract or 0.0
        ) * position.contracts_open
        financing_pnl = 0.0
        maintenance_margin_requirement = 0.0
        margin_excess = np.nan
        margin_deficit = np.nan
        in_margin_call = False
        margin_call_days = 0
        forced_liquidation = False
        contracts_liquidated = 0
        margin_status = None
        if position.margin_account is not None:
            margin_status = position.margin_account.evaluate(
                equity=equity_running + valuation.delta_pnl_market,
                initial_margin_requirement=initial_margin_requirement,
                open_contracts=position.contracts_open,
                as_of=curr_date,
            )
            financing_pnl = margin_status.financing_pnl
            maintenance_margin_requirement = (
                margin_status.maintenance_margin_requirement
            )
            margin_excess = margin_status.margin_excess
            margin_deficit = margin_status.margin_deficit
            in_margin_call = margin_status.in_margin_call
            margin_call_days = margin_status.margin_call_days
            forced_liquidation = margin_status.forced_liquidation
            contracts_liquidated = margin_status.contracts_to_liquidate

        return _MarkMarginSnapshot(
            initial_margin_requirement=initial_margin_requirement,
            financing_pnl=financing_pnl,
            maintenance_margin_requirement=maintenance_margin_requirement,
            margin_excess=margin_excess,
            margin_deficit=margin_deficit,
            in_margin_call=in_margin_call,
            margin_call_days=margin_call_days,
            forced_liquidation=forced_liquidation,
            contracts_liquidated=contracts_liquidated,
            margin_status=margin_status,
        )

    @staticmethod
    def _build_mark_record(
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        valuation: _MarkValuationSnapshot,
        margin: _MarkMarginSnapshot,
    ) -> dict:
        """Build one-date MTM record before forced close or standard exits."""
        delta_pnl = valuation.delta_pnl_market + margin.financing_pnl
        return {
            "date": curr_date,
            "S": valuation.spot,
            "iv": valuation.iv,
            "delta_pnl": delta_pnl,
            "delta": valuation.delta,
            "net_delta": valuation.net_delta,
            "gamma": valuation.gamma,
            "vega": valuation.vega,
            "theta": valuation.theta,
            "hedge_qty": position.hedge_qty,
            "hedge_price_prev": position.hedge_price_entry,
            "hedge_pnl": valuation.hedge_pnl,
            "open_contracts": position.contracts_open,
            "margin_per_contract": position.latest_margin_per_contract,
            "initial_margin_requirement": margin.initial_margin_requirement,
            "maintenance_margin_requirement": margin.maintenance_margin_requirement,
            "margin_excess": margin.margin_excess,
            "margin_deficit": margin.margin_deficit,
            "in_margin_call": margin.in_margin_call,
            "margin_call_days": margin.margin_call_days,
            "forced_liquidation": margin.forced_liquidation,
            "contracts_liquidated": margin.contracts_liquidated,
            "financing_pnl": margin.financing_pnl,
        }

    @staticmethod
    def _update_position_mark_state(
        *,
        position: OpenPosition,
        pnl_mtm: float,
        spot: float,
        iv: float,
        delta: float,
        net_delta: float,
        gamma: float,
        vega: float,
        theta: float,
    ) -> None:
        """Persist updated in-trade state after one mark step."""
        position.prev_mtm = pnl_mtm
        position.last_spot = spot
        position.last_iv = iv
        position.last_delta = delta
        position.last_net_delta = net_delta
        position.last_gamma = gamma
        position.last_vega = vega
        position.last_theta = theta

    def _exit_prices_for_position(
        self,
        *,
        position: OpenPosition,
        leg_quotes: tuple[pd.Series, ...],
        cfg: BacktestConfig,
    ) -> tuple[float, ...]:
        """Compute executable exit prices for all legs of one open position."""
        return tuple(
            self._exit_leg_price(
                quote,
                side=_effective_leg_side(leg),
                cfg=cfg,
            )
            for leg, quote in zip(position.intent.legs, leg_quotes, strict=True)
        )

    def _build_trade_row(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        contracts: int,
        pnl: float,
        exit_type: str,
        exit_prices: tuple[float, ...],
    ) -> dict:
        """Build one trade ledger row for a close/forced-liquidation action."""
        return {
            "entry_date": position.entry_date,
            "exit_date": curr_date,
            "entry_dte": position.chosen_dte,
            "expiry_date": position.expiry_date,
            "contracts": contracts,
            "pnl": pnl,
            "risk_per_contract": position.risk_per_contract,
            "risk_worst_scenario": position.risk_worst_scenario,
            "margin_per_contract": position.latest_margin_per_contract,
            "exit_type": exit_type,
            "trade_legs": self._trade_legs_payload(
                legs=position.intent.legs,
                exit_prices=exit_prices,
            ),
        }

    @staticmethod
    def _apply_closed_position_fields(
        mtm_record: dict,
        *,
        delta_pnl: float,
        equity_after: float,
    ) -> None:
        """Update MTM record fields for a fully closed position."""
        mtm_record.update(
            {
                "delta_pnl": delta_pnl,
                "delta": 0.0,
                "net_delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "hedge_qty": 0.0,
                "open_contracts": 0,
                "initial_margin_requirement": 0.0,
                "maintenance_margin_requirement": 0.0,
                "margin_excess": equity_after,
                "margin_deficit": 0.0,
                "in_margin_call": False,
            }
        )

    def _handle_forced_liquidation(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        cfg: BacktestConfig,
        lot_size: int,
        roundtrip_commission_per_contract: float,
        equity_running: float,
        valuation: _MarkValuationSnapshot,
        margin: _MarkMarginSnapshot,
        mtm_record: dict,
    ) -> tuple[OpenPosition | None, dict, list[dict]] | None:
        """Handle forced margin liquidation and return lifecycle outcome if triggered."""
        if (
            margin.margin_status is None
            or not margin.margin_status.forced_liquidation
            or margin.margin_status.contracts_to_liquidate <= 0
            or valuation.complete_leg_quotes is None
        ):
            return None

        exit_prices = self._exit_prices_for_position(
            position=position,
            leg_quotes=valuation.complete_leg_quotes,
            cfg=cfg,
        )
        contracts_to_close = margin.margin_status.contracts_to_liquidate
        contracts_after = position.contracts_open - contracts_to_close
        pnl_per_contract = self._pnl_per_contract_from_exit_prices(
            legs=position.intent.legs,
            exit_prices=exit_prices,
            lot_size=lot_size,
        )
        real_pnl_closed = pnl_per_contract * contracts_to_close + valuation.hedge_pnl
        pnl_net_closed = real_pnl_closed - (
            roundtrip_commission_per_contract * contracts_to_close
        )

        trade_row = self._build_trade_row(
            position=position,
            curr_date=curr_date,
            contracts=contracts_to_close,
            pnl=pnl_net_closed,
            exit_type=(
                "Margin Call Liquidation"
                if contracts_after == 0
                else "Margin Call Partial Liquidation"
            ),
            exit_prices=exit_prices,
        )
        trade_rows = [trade_row]

        if contracts_after == 0:
            forced_delta_pnl = (
                pnl_net_closed - valuation.prev_mtm_before
            ) + margin.financing_pnl
            equity_after = equity_running + forced_delta_pnl
            self._apply_closed_position_fields(
                mtm_record,
                delta_pnl=forced_delta_pnl,
                equity_after=equity_after,
            )
            mtm_record.update(
                {
                    "forced_liquidation": True,
                    "contracts_liquidated": contracts_to_close,
                }
            )
            return None, mtm_record, trade_rows

        ratio_remaining = contracts_after / position.contracts_open
        pnl_mtm_remaining = valuation.pnl_mtm * ratio_remaining
        forced_delta_pnl = (
            pnl_net_closed + pnl_mtm_remaining - valuation.prev_mtm_before
        ) + margin.financing_pnl
        mtm_record.update(
            {
                "delta_pnl": forced_delta_pnl,
                "delta": valuation.delta * ratio_remaining,
                "net_delta": valuation.net_delta * ratio_remaining,
                "gamma": valuation.gamma * ratio_remaining,
                "vega": valuation.vega * ratio_remaining,
                "theta": valuation.theta * ratio_remaining,
                "open_contracts": contracts_after,
                "initial_margin_requirement": (
                    (position.latest_margin_per_contract or 0.0) * contracts_after
                ),
                "maintenance_margin_requirement": (
                    (position.latest_margin_per_contract or 0.0)
                    * contracts_after
                    * (
                        self.margin_policy.maintenance_margin_ratio
                        if self.margin_policy is not None
                        else 0.0
                    )
                ),
                "contracts_liquidated": contracts_to_close,
            }
        )
        position.contracts_open = contracts_after
        position.net_entry *= ratio_remaining
        self._update_position_mark_state(
            position=position,
            pnl_mtm=pnl_mtm_remaining,
            spot=valuation.spot,
            iv=valuation.iv,
            delta=float(mtm_record["delta"]),
            net_delta=float(mtm_record["net_delta"]),
            gamma=float(mtm_record["gamma"]),
            vega=float(mtm_record["vega"]),
            theta=float(mtm_record["theta"]),
        )
        return position, mtm_record, trade_rows

    def _handle_standard_exit(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        exit_type: str,
        cfg: BacktestConfig,
        lot_size: int,
        roundtrip_commission_per_contract: float,
        equity_running: float,
        valuation: _MarkValuationSnapshot,
        margin: _MarkMarginSnapshot,
        mtm_record: dict,
    ) -> tuple[OpenPosition | None, dict, list[dict]]:
        """Close a position on explicit exit rule trigger and build lifecycle outputs."""
        assert valuation.complete_leg_quotes is not None  # nosec B101
        exit_prices = self._exit_prices_for_position(
            position=position,
            leg_quotes=valuation.complete_leg_quotes,
            cfg=cfg,
        )
        pnl_per_contract = self._pnl_per_contract_from_exit_prices(
            legs=position.intent.legs,
            exit_prices=exit_prices,
            lot_size=lot_size,
        )
        real_pnl = pnl_per_contract * position.contracts_open + valuation.hedge_pnl
        pnl_net = real_pnl - (
            roundtrip_commission_per_contract * position.contracts_open
        )
        exit_delta_pnl = (pnl_net - valuation.prev_mtm_before) + margin.financing_pnl
        equity_after = equity_running + exit_delta_pnl

        trade_row = self._build_trade_row(
            position=position,
            curr_date=curr_date,
            contracts=position.contracts_open,
            pnl=pnl_net,
            exit_type=exit_type,
            exit_prices=exit_prices,
        )

        self._apply_closed_position_fields(
            mtm_record,
            delta_pnl=exit_delta_pnl,
            equity_after=equity_after,
        )
        return None, mtm_record, [trade_row]

    def open_position(
        self,
        *,
        setup: PositionEntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition, dict]:
        """Open one position and emit its entry-day MTM accounting record."""
        contracts_open = int(setup.contracts)
        lot_size = cfg.lot_size
        roundtrip_commission_per_contract = 2 * cfg.commission_per_leg
        net_entry = self._entry_net_notional(
            legs=setup.intent.legs,
            lot_size=lot_size,
            contracts=contracts_open,
        )
        delta_pc, gamma_pc, vega_pc, theta_pc = self._greeks_per_contract(
            leg_quotes=tuple((leg, leg.quote) for leg in setup.intent.legs),
            lot_size=lot_size,
        )
        delta = delta_pc * contracts_open
        gamma = gamma_pc * contracts_open
        vega = vega_pc * contracts_open
        theta = theta_pc * contracts_open
        net_delta = delta

        margin = self._evaluate_entry_margin(
            setup=setup,
            equity_running=equity_running,
            contracts_open=contracts_open,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
        )
        entry_record = self._build_entry_record(
            setup=setup,
            contracts_open=contracts_open,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            net_delta=net_delta,
            margin=margin,
        )
        position = self._build_open_position(
            setup=setup,
            contracts_open=contracts_open,
            net_entry=net_entry,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            net_delta=net_delta,
            margin=margin,
        )
        return position, entry_record

    def mark_position(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition | None, dict, list[dict]]:
        """Revalue one open position for one date and apply exit/liquidation rules.

        Returns:
            Tuple ``(updated_position_or_none, mtm_record, trade_rows)``.
        """
        lot_size = cfg.lot_size
        roundtrip_commission_per_contract = 2 * cfg.commission_per_leg

        valuation = self._resolve_mark_valuation(
            position=position,
            curr_date=curr_date,
            options=options,
            lot_size=lot_size,
        )
        self._maybe_refresh_margin_per_contract(
            position=position,
            curr_date=curr_date,
            lot_size=lot_size,
            valuation=valuation,
        )
        margin = self._evaluate_mark_margin(
            position=position,
            curr_date=curr_date,
            equity_running=equity_running,
            valuation=valuation,
        )
        mtm_record = self._build_mark_record(
            position=position,
            curr_date=curr_date,
            valuation=valuation,
            margin=margin,
        )

        forced_outcome = self._handle_forced_liquidation(
            position=position,
            curr_date=curr_date,
            cfg=cfg,
            lot_size=lot_size,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
            equity_running=equity_running,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
        )
        if forced_outcome is not None:
            return forced_outcome

        if valuation.has_missing_quote:
            self._update_position_mark_state(
                position=position,
                pnl_mtm=valuation.pnl_mtm,
                spot=valuation.spot,
                iv=valuation.iv,
                delta=float(mtm_record["delta"]),
                net_delta=float(mtm_record["net_delta"]),
                gamma=float(mtm_record["gamma"]),
                vega=float(mtm_record["vega"]),
                theta=float(mtm_record["theta"]),
            )
            return position, mtm_record, []

        exit_type = self.exit_rule_set.evaluate(curr_date=curr_date, position=position)
        if exit_type is None:
            self._update_position_mark_state(
                position=position,
                pnl_mtm=valuation.pnl_mtm,
                spot=valuation.spot,
                iv=valuation.iv,
                delta=float(mtm_record["delta"]),
                net_delta=float(mtm_record["net_delta"]),
                gamma=float(mtm_record["gamma"]),
                vega=float(mtm_record["vega"]),
                theta=float(mtm_record["theta"]),
            )
            return position, mtm_record, []

        return self._handle_standard_exit(
            position=position,
            curr_date=curr_date,
            exit_type=exit_type,
            cfg=cfg,
            lot_size=lot_size,
            roundtrip_commission_per_contract=roundtrip_commission_per_contract,
            equity_running=equity_running,
            valuation=valuation,
            margin=margin,
            mtm_record=mtm_record,
        )
