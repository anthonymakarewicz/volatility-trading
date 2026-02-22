from dataclasses import dataclass

import numpy as np
import pandas as pd

from volatility_trading.backtesting import (
    BacktestConfig,
    MarginAccount,
    MarginPolicy,
    SliceContext,
)
from volatility_trading.filters import Filter
from volatility_trading.options import (
    BlackScholesPricer,
    FixedGridScenarioGenerator,
    MarginModel,
    PriceModel,
    RegTMarginModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
)
from volatility_trading.signals import Signal

from ..base_strategy import Strategy
from ..options_core import (
    choose_expiry_by_target_dte,
    estimate_short_straddle_margin_per_contract,
    pick_quote_by_delta,
    size_short_straddle_contracts,
    time_to_expiry_years,
)


@dataclass(frozen=True)
class _EntrySetup:
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
class _OpenShortStraddlePosition:
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


class VRPHarvestingStrategy(Strategy):
    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        rebalance_period: int | None = 5,
        max_holding_period: int | None = None,
        allow_same_day_reentry_on_rebalance: bool = True,
        allow_same_day_reentry_on_max_holding: bool = False,
        target_dte: int = 30,
        max_dte_diff: int = 7,
        pricer: PriceModel | None = None,
        scenario_generator: ScenarioGenerator | None = None,
        risk_estimator: RiskEstimator | None = None,
        risk_budget_pct: float | None = None,
        margin_model: MarginModel | None = None,
        margin_budget_pct: float | None = None,
        margin_policy: MarginPolicy | None = None,
        min_contracts: int = 1,
        max_contracts: int | None = None,
    ):
        """
        Baseline VRP harvesting strategy:
        - short ATM straddle when signal is ON
        - 30D target maturity (by dte)
        - optional rebalance cadence and max holding-age exit
        - optional stress-based risk sizing and margin-based capacity limits
        - no delta-hedge (for now)
        """
        super().__init__(signal=signal, filters=filters)
        for name, period in (
            ("rebalance_period", rebalance_period),
            ("max_holding_period", max_holding_period),
        ):
            if period is not None and period <= 0:
                raise ValueError(f"{name} must be > 0 when provided")

        self.rebalance_period = rebalance_period
        self.max_holding_period = max_holding_period
        if self.rebalance_period is None and self.max_holding_period is None:
            raise ValueError(
                "At least one of rebalance_period or max_holding_period must be set."
            )

        self.allow_same_day_reentry_on_rebalance = allow_same_day_reentry_on_rebalance
        self.allow_same_day_reentry_on_max_holding = (
            allow_same_day_reentry_on_max_holding
        )
        self.target_dte = target_dte
        self.max_dte_diff = max_dte_diff
        self.pricer = pricer or BlackScholesPricer()
        self.scenario_generator = scenario_generator or FixedGridScenarioGenerator()
        self.risk_estimator = risk_estimator or StressLossRiskEstimator()
        if risk_budget_pct is not None:
            self.risk_sizer = RiskBudgetSizer(
                risk_budget_pct=risk_budget_pct,
                min_contracts=min_contracts,
                max_contracts=max_contracts,
            )
        else:
            self.risk_sizer = None
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts

        self.margin_model = margin_model
        if self.margin_model is None and margin_budget_pct is not None:
            self.margin_model = RegTMarginModel()

        self.margin_budget_pct = margin_budget_pct
        if self.margin_model is not None and self.margin_budget_pct is None:
            self.margin_budget_pct = 1.0
        if self.margin_budget_pct is not None and not 0 <= self.margin_budget_pct <= 1:
            raise ValueError("margin_budget_pct must be in [0, 1]")
        self.margin_policy = margin_policy

    # --------- Helpers ---------

    @staticmethod
    def _pick_quote(df_leg: pd.DataFrame, tgt: float, delta_tolerance: float = 0.05):
        """
        Pick the quote whose delta is closest to `tgt` within `delta_tolerance`.
        Used here to approximate ATM (tgt ≈ +0.5 for calls, -0.5 for puts).
        """
        return pick_quote_by_delta(
            df_leg,
            target_delta=tgt,
            delta_tolerance=delta_tolerance,
        )

    @staticmethod
    def _compute_greeks_per_contract(
        put_q: pd.Series,
        call_q: pd.Series,
        put_side: int,
        call_side: int,
        lot_size: int,
    ):
        """
        Aggregate Greeks per contract for a 1-lot straddle.
        sides: +1 for long, -1 for short (per leg).
        """
        delta = (put_side * put_q["delta"] + call_side * call_q["delta"]) * lot_size
        gamma = (put_side * put_q["gamma"] + call_side * call_q["gamma"]) * lot_size
        vega = (put_side * put_q["vega"] + call_side * call_q["vega"]) * lot_size
        theta = (put_side * put_q["theta"] + call_side * call_q["theta"]) * lot_size
        return delta, gamma, vega, theta

    @staticmethod
    def _time_to_expiry_years(
        *,
        entry_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        quote_yte: float | int | None,
        quote_dte: float | int | None,
    ) -> float:
        return time_to_expiry_years(
            entry_date=entry_date,
            expiry_date=expiry_date,
            quote_yte=quote_yte,
            quote_dte=quote_dte,
        )

    def _estimate_margin_per_contract(
        self,
        *,
        as_of_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        put_quote: pd.Series,
        call_quote: pd.Series,
        put_entry: float,
        call_entry: float,
        lot_size: int,
        spot: float,
        volatility: float,
    ) -> float | None:
        """Estimate current margin requirement for one straddle contract set."""
        return estimate_short_straddle_margin_per_contract(
            as_of_date=as_of_date,
            expiry_date=expiry_date,
            put_quote=put_quote,
            call_quote=call_quote,
            put_entry=put_entry,
            call_entry=call_entry,
            lot_size=lot_size,
            spot=spot,
            volatility=volatility,
            margin_model=self.margin_model,
            pricer=self.pricer,
        )

    def _size_contracts(
        self,
        *,
        entry_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        put_q: pd.Series,
        call_q: pd.Series,
        put_entry: float,
        call_entry: float,
        lot_size: int,
        spot: float,
        volatility: float,
        equity: float,
    ) -> tuple[int, float | None, str | None, float | None]:
        """Compute contracts from risk budget and optional margin-capacity budget."""
        return size_short_straddle_contracts(
            entry_date=entry_date,
            expiry_date=expiry_date,
            put_quote=put_q,
            call_quote=call_q,
            put_entry=put_entry,
            call_entry=call_entry,
            lot_size=lot_size,
            spot=spot,
            volatility=volatility,
            equity=equity,
            pricer=self.pricer,
            scenario_generator=self.scenario_generator,
            risk_estimator=self.risk_estimator,
            risk_sizer=self.risk_sizer,
            margin_model=self.margin_model,
            margin_budget_pct=self.margin_budget_pct,
            min_contracts=self.min_contracts,
            max_contracts=self.max_contracts,
        )

    @staticmethod
    def choose_vrp_expiry(
        chain: pd.DataFrame,
        target_dte: int = 30,
        max_dte_diff: int = 7,
        min_atm_quotes: int = 2,
    ) -> int | None:
        """
        Pick a single expiry_date for the VRP straddle:
        - nearest to target_dte within ±max_dte_diff
        - with at least `min_atm_quotes` quotes near ATM.
        """
        return choose_expiry_by_target_dte(
            chain=chain,
            target_dte=target_dte,
            max_dte_diff=max_dte_diff,
            min_atm_quotes=min_atm_quotes,
        )

    # TODO: Probably would have a choose contract that woudl consider
    # both T and delta for teh choice of the contract (e.g. takign the best contarct
    # within [dte_min, dte_max] X [delta_min, delta_max])

    # --------- Main entry point ---------

    def run(self, ctx: SliceContext):
        data = ctx.data
        capital = ctx.capital
        cfg = ctx.config

        options = data["options"]
        features = data.get("features")  # expected: DataFrame with iv_atm etc.
        hedge = data.get("hedge")  # not used yet, but kept in signature

        # For now: always-on short vol (filters will switch us OFF if needed)
        series = pd.Series(0, index=options.index)
        signals = self.signal.generate_signals(series)

        # TODO: Apply the filters usign a for loop to validate signals

        trades, mtm = self._simulate_short_straddles(
            options=options,
            signals=signals,
            features=features,
            hedge=hedge,
            capital=capital,
            cfg=cfg,
        )
        return trades, mtm

    # --------- Core simulator ---------

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
    def _signals_to_on_frame(signals: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Normalize signals to a sorted DataFrame with a boolean `on` column."""
        if isinstance(signals, pd.Series):
            sig_df = signals.to_frame("on")
        else:
            sig_df = signals.copy()

        if "on" not in sig_df.columns:
            if "short" in sig_df.columns:
                sig_df["on"] = sig_df["short"].astype(bool)
            elif "long" in sig_df.columns:
                sig_df["on"] = sig_df["long"].astype(bool)
            elif sig_df.shape[1] == 1:
                sig_df["on"] = sig_df.iloc[:, 0].astype(bool)
            else:
                raise ValueError(
                    "VRPStrategy expects a boolean 'on' column in signals, "
                    "or a Series, or a DF with 'short'/'long' or a single column."
                )
        return sig_df.sort_index()

    def _prepare_entry_setup(
        self,
        *,
        entry_date: pd.Timestamp,
        options: pd.DataFrame,
        features: pd.DataFrame | None,
        equity_running: float,
        cfg: BacktestConfig,
    ) -> _EntrySetup | None:
        """Build an entry setup when all leg-selection and sizing constraints pass."""
        chain = self._chain_for_date(options, entry_date)

        chosen_dte = self.choose_vrp_expiry(
            chain=chain,
            target_dte=self.target_dte,
            max_dte_diff=self.max_dte_diff,
        )
        if chosen_dte is None:
            return None

        chain = chain[chain["dte"] == chosen_dte]
        expiry_date = pd.Timestamp(chain["expiry_date"].iloc[0])

        puts = chain[chain["option_type"] == "P"]
        calls = chain[chain["option_type"] == "C"]
        put_q = self._pick_quote(puts, tgt=-0.5, delta_tolerance=0.10)
        call_q = self._pick_quote(calls, tgt=+0.5, delta_tolerance=0.10)
        if put_q is None or call_q is None:
            return None

        put_side = -1
        call_side = -1
        put_entry = float(put_q["bid_price"] - cfg.slip_bid)
        call_entry = float(call_q["bid_price"] - cfg.slip_bid)
        delta_pc, gamma_pc, vega_pc, theta_pc = self._compute_greeks_per_contract(
            put_q, call_q, put_side, call_side, cfg.lot_size
        )

        spot_entry = float(chain["spot_price"].iloc[0])
        if "smoothed_iv" in put_q.index and "smoothed_iv" in call_q.index:
            iv_entry = float(0.5 * (put_q["smoothed_iv"] + call_q["smoothed_iv"]))
        elif (
            features is not None
            and entry_date in features.index
            and "iv_atm" in features.columns
        ):
            iv_entry = float(features.loc[entry_date, "iv_atm"])
        else:
            iv_entry = float("nan")

        contracts, risk_pc, risk_scenario, margin_pc = self._size_contracts(
            entry_date=entry_date,
            expiry_date=expiry_date,
            put_q=put_q,
            call_q=call_q,
            put_entry=put_entry,
            call_entry=call_entry,
            lot_size=cfg.lot_size,
            spot=spot_entry,
            volatility=iv_entry,
            equity=float(equity_running),
        )
        if contracts <= 0:
            return None

        return _EntrySetup(
            entry_date=entry_date,
            expiry_date=expiry_date,
            chosen_dte=int(chosen_dte),
            put_q=put_q,
            call_q=call_q,
            put_side=put_side,
            call_side=call_side,
            put_entry=put_entry,
            call_entry=call_entry,
            spot_entry=spot_entry,
            iv_entry=iv_entry,
            delta_per_contract=float(delta_pc),
            gamma_per_contract=float(gamma_pc),
            vega_per_contract=float(vega_pc),
            theta_per_contract=float(theta_pc),
            contracts=int(contracts),
            risk_per_contract=risk_pc,
            risk_worst_scenario=risk_scenario,
            margin_per_contract=margin_pc,
        )

    def _open_position(
        self,
        *,
        setup: _EntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[_OpenShortStraddlePosition, dict]:
        """Create one open-position state and its entry-day MTM record."""
        contracts_open = int(setup.contracts)
        lot_size = cfg.lot_size
        roundtrip_comm_pc = (
            2 * cfg.commission_per_leg
        )  # TODO: Should sclae by Options in the strcuture

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

        position = _OpenShortStraddlePosition(
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

    def _mark_open_position(
        self,
        *,
        position: _OpenShortStraddlePosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[_OpenShortStraddlePosition | None, dict, list[dict]]:
        """Revalue one open position on one date and apply exits/liquidations.

        This function centralizes position lifecycle business logic:
        - MTM with carry-forward when quotes are missing
        - financing and maintenance checks
        - forced liquidation (full/partial) and holding-period exit
        """
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
            iv_curr = float(  # TODO: Use mid_iv instead and use a vega for eahc IV as a separate risk
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
            margin_pc_curr = self._estimate_margin_per_contract(
                as_of_date=curr_date,
                expiry_date=position.expiry_date,
                put_quote=pt,
                call_quote=ct,
                put_entry=float(position.put_entry),
                call_entry=float(position.call_entry),
                lot_size=lot_size,
                spot=float(S_curr),
                volatility=float(iv_curr),
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
                # Full liquidation replaces today's open-MTM delta with realized closeout.
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
            # TODO: For latter chekc if all options in the structure ha value quote
            # For the ones where thre is a quote update the MTM accoridngly for the others
            # assume no changes

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

    def _can_reenter_same_day(self, trade_rows: list[dict]) -> bool:
        """Allow same-day reentry according to exit-specific policy."""
        for row in trade_rows:
            exit_type = row.get("exit_type")
            if exit_type == "Rebalance Period":
                return self.allow_same_day_reentry_on_rebalance
            if exit_type == "Max Holding Period":
                return self.allow_same_day_reentry_on_max_holding
            if exit_type == "Rebalance/Max Holding Period":
                return (
                    self.allow_same_day_reentry_on_rebalance
                    or self.allow_same_day_reentry_on_max_holding
                )
        return False

    def _simulate_short_straddles(
        self,
        options: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None,
        hedge: pd.Series | None,
        capital: float,
        cfg: BacktestConfig,
    ):
        """Run a single-position-at-a-time, date-driven short-straddle simulation."""
        _ = hedge

        sig_df = self._signals_to_on_frame(signals)
        options = options.sort_index()
        trading_dates = sorted(options.index.unique())
        active_signal_dates = set(sig_df.index[sig_df["on"]])

        trades: list[dict] = []
        mtm_records: list[dict] = []
        equity_running = float(capital)
        open_position: _OpenShortStraddlePosition | None = None

        for curr_date in trading_dates:
            if open_position is not None:
                open_position, mtm_record, trade_rows = self._mark_open_position(
                    position=open_position,
                    curr_date=curr_date,
                    options=options,
                    cfg=cfg,
                    equity_running=equity_running,
                )
                mtm_records.append(mtm_record)
                trades.extend(trade_rows)
                equity_running += float(mtm_record["delta_pnl"])

                if open_position is not None:
                    continue
                if not self._can_reenter_same_day(trade_rows):
                    continue

            if curr_date not in active_signal_dates:
                continue

            setup = self._prepare_entry_setup(
                entry_date=curr_date,
                options=options,
                features=features,
                equity_running=equity_running,
                cfg=cfg,
            )
            if setup is None:
                continue

            open_position, entry_record = self._open_position(
                setup=setup,
                cfg=cfg,
                equity_running=equity_running,
            )
            mtm_records.append(entry_record)
            equity_running += float(entry_record["delta_pnl"])

        if not mtm_records:
            return pd.DataFrame(trades), pd.DataFrame()

        mtm_agg = pd.DataFrame(mtm_records).set_index("date").sort_index()
        agg_map = {
            "delta_pnl": "sum",
            "delta": "sum",
            "net_delta": "sum",
            "gamma": "sum",
            "vega": "sum",
            "theta": "sum",
            "hedge_pnl": "sum",
            "S": "first",
            "iv": "first",
        }
        optional_sum_cols = [
            "open_contracts",
            "initial_margin_requirement",
            "maintenance_margin_requirement",
            "contracts_liquidated",
            "financing_pnl",
        ]
        optional_first_cols = [
            "margin_per_contract",
            "margin_excess",
            "margin_deficit",
        ]
        optional_max_cols = ["in_margin_call", "margin_call_days", "forced_liquidation"]
        for col in optional_sum_cols:
            if col in mtm_agg.columns:
                agg_map[col] = "sum"
        for col in optional_first_cols:
            if col in mtm_agg.columns:
                agg_map[col] = "first"
        for col in optional_max_cols:
            if col in mtm_agg.columns:
                agg_map[col] = "max"
        mtm = mtm_agg.groupby("date").agg(agg_map)
        mtm["equity"] = capital + mtm["delta_pnl"].cumsum()

        return pd.DataFrame(trades), mtm
