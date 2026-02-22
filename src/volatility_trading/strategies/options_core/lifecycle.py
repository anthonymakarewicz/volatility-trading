"""Shared open/mark/close lifecycle engine for short-straddle positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volatility_trading.backtesting import (
    BacktestConfig,
    MarginAccount,
    MarginPolicy,
)
from volatility_trading.options import MarginModel, PriceModel

from .sizing import estimate_short_straddle_margin_per_contract


@dataclass(frozen=True)
class ShortStraddleEntrySetup:
    """Snapshot of all entry-time decisions needed to open one position."""

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    chosen_dte: int
    put_q: pd.Series
    call_q: pd.Series
    put_side: int
    call_side: int
    put_entry: float
    call_entry: float
    spot_entry: float
    iv_entry: float
    delta_per_contract: float
    gamma_per_contract: float
    vega_per_contract: float
    theta_per_contract: float
    contracts: int
    risk_per_contract: float | None
    risk_worst_scenario: str | None
    margin_per_contract: float | None


@dataclass
class OpenShortStraddlePosition:
    """Mutable open-position state updated once per trading date."""

    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    chosen_dte: int
    rebalance_date: pd.Timestamp | None
    max_hold_date: pd.Timestamp | None
    put_q: pd.Series
    call_q: pd.Series
    put_side: int
    call_side: int
    put_entry: float
    call_entry: float
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
class ShortStraddleLifecycleEngine:
    """Shared position lifecycle logic: open, mark, and close."""

    rebalance_period: int | None
    max_holding_period: int | None
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
    def _compute_greeks_per_contract(
        put_q: pd.Series,
        call_q: pd.Series,
        put_side: int,
        call_side: int,
        lot_size: int,
    ) -> tuple[float, float, float, float]:
        """Aggregate Greeks per contract for a 1-lot straddle."""
        delta = (put_side * put_q["delta"] + call_side * call_q["delta"]) * lot_size
        gamma = (put_side * put_q["gamma"] + call_side * call_q["gamma"]) * lot_size
        vega = (put_side * put_q["vega"] + call_side * call_q["vega"]) * lot_size
        theta = (put_side * put_q["theta"] + call_side * call_q["theta"]) * lot_size
        return float(delta), float(gamma), float(vega), float(theta)

    def open_position(
        self,
        *,
        setup: ShortStraddleEntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenShortStraddlePosition, dict]:
        """Create one open-position state and its entry-day MTM record."""
        contracts_open = int(setup.contracts)
        lot_size = cfg.lot_size
        roundtrip_comm_pc = 2 * cfg.commission_per_leg

        net_entry = (
            (setup.put_side * setup.put_entry + setup.call_side * setup.call_entry)
            * lot_size
            * contracts_open
        )
        delta = setup.delta_per_contract * contracts_open
        gamma = setup.gamma_per_contract * contracts_open
        vega = setup.vega_per_contract * contracts_open
        theta = setup.theta_per_contract * contracts_open
        net_delta = delta

        margin_account = (
            MarginAccount(self.margin_policy) if self.margin_policy else None
        )
        latest_margin_pc = setup.margin_per_contract
        initial_margin_req = (latest_margin_pc or 0.0) * contracts_open
        entry_delta_pnl = -(roundtrip_comm_pc * contracts_open)
        entry_financing = 0.0
        maintenance_margin_req = 0.0
        margin_excess = np.nan
        margin_deficit = np.nan
        in_margin_call = False
        margin_call_days = 0
        forced_liquidation = False
        contracts_liquidated = 0

        if margin_account is not None:
            margin_status = margin_account.evaluate(
                equity=equity_running + entry_delta_pnl,
                initial_margin_requirement=initial_margin_req,
                open_contracts=contracts_open,
                as_of=setup.entry_date,
            )
            entry_financing = margin_status.financing_pnl
            entry_delta_pnl += entry_financing
            maintenance_margin_req = margin_status.maintenance_margin_requirement
            margin_excess = margin_status.margin_excess
            margin_deficit = margin_status.margin_deficit
            in_margin_call = margin_status.in_margin_call
            margin_call_days = margin_status.margin_call_days
            forced_liquidation = margin_status.forced_liquidation
            contracts_liquidated = margin_status.contracts_to_liquidate

        entry_record = {
            "date": setup.entry_date,
            "S": setup.spot_entry,
            "iv": setup.iv_entry,
            "delta_pnl": entry_delta_pnl,
            "delta": delta,
            "net_delta": net_delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "hedge_qty": 0.0,
            "hedge_price_prev": np.nan,
            "hedge_pnl": 0.0,
            "open_contracts": contracts_open,
            "margin_per_contract": latest_margin_pc,
            "initial_margin_requirement": initial_margin_req,
            "maintenance_margin_requirement": maintenance_margin_req,
            "margin_excess": margin_excess,
            "margin_deficit": margin_deficit,
            "in_margin_call": in_margin_call,
            "margin_call_days": margin_call_days,
            "forced_liquidation": forced_liquidation,
            "contracts_liquidated": contracts_liquidated,
            "financing_pnl": entry_financing,
        }

        rebalance_date = (
            setup.entry_date + pd.Timedelta(days=self.rebalance_period)
            if self.rebalance_period is not None
            else None
        )
        max_hold_date = (
            setup.entry_date + pd.Timedelta(days=self.max_holding_period)
            if self.max_holding_period is not None
            else None
        )

        position = OpenShortStraddlePosition(
            entry_date=setup.entry_date,
            expiry_date=setup.expiry_date,
            chosen_dte=setup.chosen_dte,
            rebalance_date=rebalance_date,
            max_hold_date=max_hold_date,
            put_q=setup.put_q,
            call_q=setup.call_q,
            put_side=setup.put_side,
            call_side=setup.call_side,
            put_entry=setup.put_entry,
            call_entry=setup.call_entry,
            contracts_open=contracts_open,
            risk_per_contract=setup.risk_per_contract,
            risk_worst_scenario=setup.risk_worst_scenario,
            margin_account=margin_account,
            latest_margin_per_contract=latest_margin_pc,
            net_entry=net_entry,
            prev_mtm=0.0,
            hedge_qty=0.0,
            hedge_price_entry=np.nan,
            last_spot=setup.spot_entry,
            last_iv=setup.iv_entry,
            last_delta=delta,
            last_gamma=gamma,
            last_vega=vega,
            last_theta=theta,
            last_net_delta=net_delta,
        )
        return position, entry_record

    def mark_position(
        self,
        *,
        position: OpenShortStraddlePosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenShortStraddlePosition | None, dict, list[dict]]:
        """Revalue one open position for one date and apply lifecycle exits."""
        lot_size = cfg.lot_size
        roundtrip_comm_pc = 2 * cfg.commission_per_leg

        chain_all = self._chain_for_date(options, curr_date)
        today_chain = chain_all[chain_all["expiry_date"] == position.expiry_date]
        put_today = today_chain[
            (today_chain["option_type"] == "P")
            & (today_chain["strike"] == position.put_q["strike"])
        ]
        call_today = today_chain[
            (today_chain["option_type"] == "C")
            & (today_chain["strike"] == position.call_q["strike"])
        ]

        prev_mtm_before = position.prev_mtm
        if put_today.empty or call_today.empty:
            pnl_mtm = prev_mtm_before
            delta = position.last_delta
            gamma = position.last_gamma
            vega = position.last_vega
            theta = position.last_theta
            pt = None
            ct = None
        else:
            pt = put_today.iloc[0]
            ct = call_today.iloc[0]
            put_mid_t = 0.5 * (pt["bid_price"] + pt["ask_price"])
            call_mid_t = 0.5 * (ct["bid_price"] + ct["ask_price"])
            pe_mid = position.put_side * put_mid_t
            ce_mid = position.call_side * call_mid_t
            pnl_mtm = (
                pe_mid + ce_mid
            ) * lot_size * position.contracts_open - position.net_entry

            delta_pc_t, gamma_pc_t, vega_pc_t, theta_pc_t = (
                self._compute_greeks_per_contract(
                    pt,
                    ct,
                    position.put_side,
                    position.call_side,
                    lot_size,
                )
            )
            delta = delta_pc_t * position.contracts_open
            gamma = gamma_pc_t * position.contracts_open
            vega = vega_pc_t * position.contracts_open
            theta = theta_pc_t * position.contracts_open

        S_curr = position.last_spot
        if "spot_price" in chain_all.columns and not chain_all.empty:
            S_curr = float(chain_all["spot_price"].iloc[0])

        iv_curr = position.last_iv
        if (
            "smoothed_iv" in today_chain.columns
            and not put_today.empty
            and not call_today.empty
        ):
            iv_curr = float(
                0.5
                * (
                    float(put_today.iloc[0]["smoothed_iv"])
                    + float(call_today.iloc[0]["smoothed_iv"])
                )
            )

        hedge_pnl = 0.0
        net_delta = delta
        delta_pnl_market = (pnl_mtm - prev_mtm_before) + hedge_pnl

        if (
            self.margin_model is not None
            and pt is not None
            and ct is not None
            and np.isfinite(iv_curr)
            and iv_curr > 0
        ):
            margin_pc_curr = estimate_short_straddle_margin_per_contract(
                as_of_date=curr_date,
                expiry_date=position.expiry_date,
                put_quote=pt,
                call_quote=ct,
                put_entry=float(position.put_entry),
                call_entry=float(position.call_entry),
                lot_size=lot_size,
                spot=float(S_curr),
                volatility=float(iv_curr),
                margin_model=self.margin_model,
                pricer=self.pricer,
            )
            if margin_pc_curr is not None:
                position.latest_margin_per_contract = margin_pc_curr

        initial_margin_req = (
            position.latest_margin_per_contract or 0.0
        ) * position.contracts_open
        financing_pnl = 0.0
        maintenance_margin_req = 0.0
        margin_excess = np.nan
        margin_deficit = np.nan
        in_margin_call = False
        margin_call_days = 0
        forced_liquidation = False
        contracts_liquidated = 0
        margin_status = None
        if position.margin_account is not None:
            margin_status = position.margin_account.evaluate(
                equity=equity_running + delta_pnl_market,
                initial_margin_requirement=initial_margin_req,
                open_contracts=position.contracts_open,
                as_of=curr_date,
            )
            financing_pnl = margin_status.financing_pnl
            maintenance_margin_req = margin_status.maintenance_margin_requirement
            margin_excess = margin_status.margin_excess
            margin_deficit = margin_status.margin_deficit
            in_margin_call = margin_status.in_margin_call
            margin_call_days = margin_status.margin_call_days
            forced_liquidation = margin_status.forced_liquidation
            contracts_liquidated = margin_status.contracts_to_liquidate

        delta_pnl = delta_pnl_market + financing_pnl
        mtm_record = {
            "date": curr_date,
            "S": S_curr,
            "iv": iv_curr,
            "delta_pnl": delta_pnl,
            "delta": delta,
            "net_delta": net_delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "hedge_qty": position.hedge_qty,
            "hedge_price_prev": position.hedge_price_entry,
            "hedge_pnl": hedge_pnl,
            "open_contracts": position.contracts_open,
            "margin_per_contract": position.latest_margin_per_contract,
            "initial_margin_requirement": initial_margin_req,
            "maintenance_margin_requirement": maintenance_margin_req,
            "margin_excess": margin_excess,
            "margin_deficit": margin_deficit,
            "in_margin_call": in_margin_call,
            "margin_call_days": margin_call_days,
            "forced_liquidation": forced_liquidation,
            "contracts_liquidated": contracts_liquidated,
            "financing_pnl": financing_pnl,
        }

        trade_rows: list[dict] = []
        if (
            margin_status is not None
            and margin_status.forced_liquidation
            and margin_status.contracts_to_liquidate > 0
            and pt is not None
            and ct is not None
        ):
            put_exit = float(pt["ask_price"] + cfg.slip_ask)
            call_exit = float(ct["ask_price"] + cfg.slip_ask)
            contracts_to_close = margin_status.contracts_to_liquidate
            contracts_after = position.contracts_open - contracts_to_close
            pnl_per_contract = (
                position.put_side * (put_exit - position.put_entry)
                + position.call_side * (call_exit - position.call_entry)
            ) * lot_size
            real_pnl_closed = pnl_per_contract * contracts_to_close + hedge_pnl
            pnl_net_closed = real_pnl_closed - (roundtrip_comm_pc * contracts_to_close)

            trade_rows.append(
                {
                    "entry_date": position.entry_date,
                    "exit_date": curr_date,
                    "entry_dte": position.chosen_dte,
                    "expiry_date": position.expiry_date,
                    "contracts": contracts_to_close,
                    "put_strike": position.put_q["strike"],
                    "call_strike": position.call_q["strike"],
                    "put_entry": position.put_entry,
                    "call_entry": position.call_entry,
                    "put_exit": put_exit,
                    "call_exit": call_exit,
                    "pnl": pnl_net_closed,
                    "risk_per_contract": position.risk_per_contract,
                    "risk_worst_scenario": position.risk_worst_scenario,
                    "margin_per_contract": position.latest_margin_per_contract,
                    "exit_type": (
                        "Margin Call Liquidation"
                        if contracts_after == 0
                        else "Margin Call Partial Liquidation"
                    ),
                }
            )

            if contracts_after == 0:
                forced_delta_pnl = (pnl_net_closed - prev_mtm_before) + financing_pnl
                equity_after = equity_running + forced_delta_pnl
                mtm_record.update(
                    {
                        "delta_pnl": forced_delta_pnl,
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
                        "forced_liquidation": True,
                        "contracts_liquidated": contracts_to_close,
                    }
                )
                return None, mtm_record, trade_rows

            ratio_remaining = contracts_after / position.contracts_open
            pnl_mtm_remaining = pnl_mtm * ratio_remaining
            forced_delta_pnl = (
                pnl_net_closed + pnl_mtm_remaining - prev_mtm_before
            ) + financing_pnl
            mtm_record.update(
                {
                    "delta_pnl": forced_delta_pnl,
                    "delta": delta * ratio_remaining,
                    "net_delta": net_delta * ratio_remaining,
                    "gamma": gamma * ratio_remaining,
                    "vega": vega * ratio_remaining,
                    "theta": theta * ratio_remaining,
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
            position.prev_mtm = pnl_mtm_remaining
            position.last_spot = S_curr
            position.last_iv = iv_curr
            position.last_delta = float(mtm_record["delta"])
            position.last_net_delta = float(mtm_record["net_delta"])
            position.last_gamma = float(mtm_record["gamma"])
            position.last_vega = float(mtm_record["vega"])
            position.last_theta = float(mtm_record["theta"])
            return position, mtm_record, trade_rows

        if put_today.empty or call_today.empty:
            position.prev_mtm = pnl_mtm
            position.last_spot = S_curr
            position.last_iv = iv_curr
            position.last_delta = float(mtm_record["delta"])
            position.last_net_delta = float(mtm_record["net_delta"])
            position.last_gamma = float(mtm_record["gamma"])
            position.last_vega = float(mtm_record["vega"])
            position.last_theta = float(mtm_record["theta"])
            return position, mtm_record, trade_rows

        rebalance_due = (
            position.rebalance_date is not None and curr_date >= position.rebalance_date
        )
        max_holding_due = (
            position.max_hold_date is not None and curr_date >= position.max_hold_date
        )
        if not rebalance_due and not max_holding_due:
            position.prev_mtm = pnl_mtm
            position.last_spot = S_curr
            position.last_iv = iv_curr
            position.last_delta = float(mtm_record["delta"])
            position.last_net_delta = float(mtm_record["net_delta"])
            position.last_gamma = float(mtm_record["gamma"])
            position.last_vega = float(mtm_record["vega"])
            position.last_theta = float(mtm_record["theta"])
            return position, mtm_record, trade_rows

        if rebalance_due and max_holding_due:
            exit_type = "Rebalance/Max Holding Period"
        elif rebalance_due:
            exit_type = "Rebalance Period"
        else:
            exit_type = "Max Holding Period"

        pt_exit = put_today.iloc[0]
        ct_exit = call_today.iloc[0]
        put_exit = float(pt_exit["ask_price"] + cfg.slip_ask)
        call_exit = float(ct_exit["ask_price"] + cfg.slip_ask)
        pnl_per_contract = (
            position.put_side * (put_exit - position.put_entry)
            + position.call_side * (call_exit - position.call_entry)
        ) * lot_size
        real_pnl = pnl_per_contract * position.contracts_open + hedge_pnl
        pnl_net = real_pnl - (roundtrip_comm_pc * position.contracts_open)
        exit_delta_pnl = (pnl_net - prev_mtm_before) + financing_pnl
        equity_after = equity_running + exit_delta_pnl

        trade_rows.append(
            {
                "entry_date": position.entry_date,
                "exit_date": curr_date,
                "entry_dte": position.chosen_dte,
                "expiry_date": position.expiry_date,
                "contracts": position.contracts_open,
                "put_strike": position.put_q["strike"],
                "call_strike": position.call_q["strike"],
                "put_entry": position.put_entry,
                "call_entry": position.call_entry,
                "put_exit": put_exit,
                "call_exit": call_exit,
                "pnl": pnl_net,
                "risk_per_contract": position.risk_per_contract,
                "risk_worst_scenario": position.risk_worst_scenario,
                "margin_per_contract": position.latest_margin_per_contract,
                "exit_type": exit_type,
            }
        )
        mtm_record.update(
            {
                "delta_pnl": exit_delta_pnl,
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
        return None, mtm_record, trade_rows
