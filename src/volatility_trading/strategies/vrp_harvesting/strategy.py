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
    MarketState,
    OptionLeg,
    OptionSpec,
    OptionType,
    PositionSide,
    PriceModel,
    RegTMarginModel,
    RiskBudgetSizer,
    RiskEstimator,
    ScenarioGenerator,
    StressLossRiskEstimator,
    contracts_for_risk_budget,
)
from volatility_trading.signals import Signal

from ..base_strategy import Strategy

# TODO(feature): Trading account with monye invetsed into both the startegy and the one used
# putnin the cash account


class VRPHarvestingStrategy(Strategy):
    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        holding_period: int = 5,
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
        - optional stress-based risk sizing and margin-based capacity limits
        - no delta-hedge (for now)
        """
        super().__init__(signal=signal, filters=filters)
        self.holding_period = holding_period
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
        df2 = df_leg.copy()
        df2["d_err"] = (df2["delta"] - tgt).abs()
        df2 = df2[df2["d_err"] <= delta_tolerance]
        if df2.empty:
            return None
        return df2.iloc[df2["d_err"].values.argmin()]

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
        def _positive_or_none(value: float | int | None) -> float | None:
            if value is None or pd.isna(value):
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric) or numeric <= 0:
                return None
            return numeric

        yte = _positive_or_none(quote_yte)
        if yte is not None:
            return yte

        dte = _positive_or_none(quote_dte)
        if dte is not None:
            return max(dte / 365.0, 1e-8)

        days = (pd.Timestamp(expiry_date) - pd.Timestamp(entry_date)).days
        return max(days / 365.0, 1e-8)

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
        if self.margin_model is None:
            return None
        if not np.isfinite(spot) or spot <= 0:
            return None
        if not np.isfinite(volatility) or volatility <= 0:
            return None

        put_spec = OptionSpec(
            strike=float(put_quote["strike"]),
            time_to_expiry=self._time_to_expiry_years(
                entry_date=as_of_date,
                expiry_date=expiry_date,
                quote_yte=put_quote.get("yte"),
                quote_dte=put_quote.get("dte"),
            ),
            option_type=OptionType.PUT,
        )
        call_spec = OptionSpec(
            strike=float(call_quote["strike"]),
            time_to_expiry=self._time_to_expiry_years(
                entry_date=as_of_date,
                expiry_date=expiry_date,
                quote_yte=call_quote.get("yte"),
                quote_dte=call_quote.get("dte"),
            ),
            option_type=OptionType.CALL,
        )
        legs = (
            OptionLeg(
                spec=put_spec,
                entry_price=float(put_entry),
                side=PositionSide.SHORT,
                contract_multiplier=float(lot_size),
            ),
            OptionLeg(
                spec=call_spec,
                entry_price=float(call_entry),
                side=PositionSide.SHORT,
                contract_multiplier=float(lot_size),
            ),
        )
        state = MarketState(spot=float(spot), volatility=float(volatility))
        return self.margin_model.initial_margin_requirement(
            legs=legs,
            state=state,
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
        invalid_market = (
            not np.isfinite(spot)
            or spot <= 0
            or not np.isfinite(volatility)
            or volatility <= 0
        )
        if invalid_market:
            fallback = self.risk_sizer.min_contracts if self.risk_sizer else 1
            return fallback, None, None, None

        put_spec = OptionSpec(
            strike=float(put_q["strike"]),
            time_to_expiry=self._time_to_expiry_years(
                entry_date=entry_date,
                expiry_date=expiry_date,
                quote_yte=put_q.get("yte"),
                quote_dte=put_q.get("dte"),
            ),
            option_type=OptionType.PUT,
        )
        call_spec = OptionSpec(
            strike=float(call_q["strike"]),
            time_to_expiry=self._time_to_expiry_years(
                entry_date=entry_date,
                expiry_date=expiry_date,
                quote_yte=call_q.get("yte"),
                quote_dte=call_q.get("dte"),
            ),
            option_type=OptionType.CALL,
        )
        legs = (
            OptionLeg(
                spec=put_spec,
                entry_price=float(put_entry),
                side=PositionSide.SHORT,
                contract_multiplier=float(lot_size),
            ),
            OptionLeg(
                spec=call_spec,
                entry_price=float(call_entry),
                side=PositionSide.SHORT,
                contract_multiplier=float(lot_size),
            ),
        )
        state = MarketState(spot=float(spot), volatility=float(volatility))
        risk_contracts: int | None = None
        risk_per_contract: float | None = None
        risk_scenario: str | None = None

        if self.risk_sizer is not None:
            scenarios = self.scenario_generator.generate(spec=call_spec, state=state)
            if not scenarios:
                risk_contracts = self.risk_sizer.min_contracts
            else:
                risk_result = self.risk_estimator.estimate_risk_per_contract(
                    legs=legs,
                    state=state,
                    scenarios=scenarios,
                    pricer=self.pricer,
                )
                risk_contracts = self.risk_sizer.size(
                    equity=equity,
                    risk_per_contract=risk_result.worst_loss,
                )
                risk_per_contract = risk_result.worst_loss
                risk_scenario = risk_result.worst_scenario.name

        margin_contracts: int | None = None
        margin_per_contract: float | None = None
        if self.margin_model is not None:
            margin_per_contract = self.margin_model.initial_margin_requirement(
                legs=legs,
                state=state,
                pricer=self.pricer,
            )
            margin_budget_pct = self.margin_budget_pct or 1.0
            margin_contracts = contracts_for_risk_budget(
                equity=equity,
                risk_budget_pct=margin_budget_pct,
                risk_per_contract=margin_per_contract,
                min_contracts=0,
                max_contracts=self.max_contracts,
            )

        if risk_contracts is not None and margin_contracts is not None:
            contracts = min(risk_contracts, margin_contracts)
        elif risk_contracts is not None:
            contracts = risk_contracts
        elif margin_contracts is not None:
            contracts = margin_contracts
        else:
            contracts = max(self.min_contracts, 1)
            if self.max_contracts is not None:
                contracts = min(contracts, self.max_contracts)

        return contracts, risk_per_contract, risk_scenario, margin_per_contract

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
        exps = chain["dte"].dropna().unique()
        exps = [d for d in exps if abs(d - target_dte) <= max_dte_diff]
        if not exps:
            return None

        def is_viable(dte):
            sub = chain[chain["dte"] == dte]
            # very simple ATM band example: |S/K - 1| <= 2%
            # assumes you have 'S' or 'spot_price'
            S = sub["spot_price"].iloc[0]
            atm_band = (sub["strike"] / S).between(0.98, 1.02)
            atm_quotes = sub[atm_band]
            return len(atm_quotes) >= min_atm_quotes

        viable = [d for d in exps if is_viable(d)]
        if not viable:
            return None

        return min(viable, key=lambda d: abs(d - target_dte))

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

    def _simulate_short_straddles(
        self,
        options: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame,
        hedge: pd.Series | None,
        capital: float,
        cfg: BacktestConfig,
    ):
        """
        Baseline backtest:
        - Short ATM straddle when signal is ON.
        - No hedging.
        - Holding-period exits.
        - Contract count optionally sized by stress risk.
        """

        lot_size = cfg.lot_size
        roundtrip_comm_pc = 2 * cfg.commission_per_leg

        # --- Normalize signals to a DataFrame with at least 'on' column ---
        if isinstance(signals, pd.Series):
            sig_df = signals.to_frame("on")
        else:
            sig_df = signals.copy()

        if "on" not in sig_df.columns:
            # heuristic: use 'short' if present, else first column
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

        sig_df = sig_df.sort_index()
        options = options.sort_index()

        trades = []
        mtm_records = []
        equity_running = float(capital)
        block_until = None

        for entry_date, row in sig_df.iterrows():
            if not row["on"]:
                continue
            if entry_date not in options.index:
                continue
            if block_until is not None and entry_date < block_until:
                continue

            chain = options.loc[entry_date]

            # --- choose expiry_date closest to target DTE if available ---
            chain = options.loc[entry_date]
            chosen_dte = self.choose_vrp_expiry(
                chain=chain, target_dte=self.target_dte, max_dte_diff=self.max_dte_diff
            )
            if chosen_dte is None:
                continue

            chain = chain[chain["dte"] == chosen_dte]
            expiry_date = chain["expiry_date"].iloc[0]

            # --- pick ATM-ish put and call using delta  ~ +/- 0.5 ---
            puts = chain[chain["option_type"] == "P"]
            calls = chain[chain["option_type"] == "C"]

            put_q = self._pick_quote(puts, tgt=-0.5, delta_tolerance=0.10)
            call_q = self._pick_quote(calls, tgt=+0.5, delta_tolerance=0.10)
            if put_q is None or call_q is None:
                continue

            # short straddle: short put & short call
            put_side = -1
            call_side = -1

            put_bid = put_q["bid_price"]
            call_bid = call_q["bid_price"]
            put_entry = put_bid - cfg.slip_bid
            call_entry = call_bid - cfg.slip_bid

            # Greeks per contract
            delta_pc, gamma_pc, vega_pc, theta_pc = self._compute_greeks_per_contract(
                put_q, call_q, put_side, call_side, lot_size
            )

            S_entry = chain["spot_price"].iloc[0]
            # Best: use IV from the actual legs if you have it
            if "smoothed_iv" in put_q.index and "smoothed_iv" in call_q.index:
                iv_entry = 0.5 * (put_q["smoothed_iv"] + call_q["smoothed_iv"])
            elif (
                features is not None
                and entry_date in features.index
                and "iv_atm" in features.columns
            ):
                iv_entry = features.loc[entry_date, "iv_atm"]
            else:
                iv_entry = np.nan

            current_equity = capital + sum(
                rec["delta_pnl"] for rec in mtm_records
            )  # TODO: maybe calculate it after each trade
            contracts, risk_pc, risk_scenario, margin_pc = self._size_contracts(
                entry_date=entry_date,
                expiry_date=expiry_date,
                put_q=put_q,
                call_q=call_q,
                put_entry=float(put_entry),
                call_entry=float(call_entry),
                lot_size=lot_size,
                spot=float(S_entry),
                volatility=float(iv_entry),
                equity=float(current_equity),
            )
            if contracts <= 0:
                continue
            contracts_open = int(contracts)

            # net entry value with sign convention (short)
            net_entry = (
                (put_side * put_entry + call_side * call_entry)
                * lot_size
                * contracts_open
            )

            # initial Greeks across all contracts
            delta = delta_pc * contracts_open
            gamma = gamma_pc * contracts_open
            vega = vega_pc * contracts_open
            theta = theta_pc * contracts_open

            # no hedge in baseline
            hedge_qty = 0
            hedge_price_entry = np.nan
            net_delta = delta
            margin_account = (
                MarginAccount(self.margin_policy) if self.margin_policy else None
            )
            latest_margin_pc = margin_pc
            initial_margin_req = (latest_margin_pc or 0.0) * contracts_open
            entry_delta_pnl = -(roundtrip_comm_pc * contracts_open)  # pay entry comm
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

            mtm_records.append(
                {
                    "date": entry_date,
                    "S": S_entry,
                    "iv": iv_entry,
                    "delta_pnl": entry_delta_pnl,
                    "delta": delta,
                    "net_delta": net_delta,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "hedge_qty": hedge_qty,
                    "hedge_price_prev": hedge_price_entry,
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
            )
            equity_running += entry_delta_pnl
            prev_mtm = 0.0  # value-based MTM; delta_pnl tracks realized increments

            hold_date = entry_date + pd.Timedelta(days=self.holding_period)
            block_until = hold_date
            future_dates = sorted(options.index[options.index > entry_date].unique())
            exited = False

            for curr_date in future_dates:
                today_chain = options.loc[curr_date]
                today_chain = today_chain[today_chain["expiry_date"] == expiry_date]

                put_today = today_chain[
                    (today_chain["option_type"] == "P")
                    & (today_chain["strike"] == put_q["strike"])
                ]
                call_today = today_chain[
                    (today_chain["option_type"] == "C")
                    & (today_chain["strike"] == call_q["strike"])
                ]

                prev_mtm_before = prev_mtm
                last_rec = mtm_records[-1]

                if put_today.empty or call_today.empty:
                    # carry last Greeks; no MTM update from options
                    pnl_mtm = prev_mtm_before
                    delta = last_rec["delta"]
                    gamma = last_rec["gamma"]
                    vega = last_rec["vega"]
                    theta = last_rec["theta"]
                    pt = None
                    ct = None
                else:
                    # recompute MTM
                    pt = put_today.iloc[0]
                    ct = call_today.iloc[0]

                    put_mid_t = 0.5 * (pt["bid_price"] + pt["ask_price"])
                    call_mid_t = 0.5 * (ct["bid_price"] + ct["ask_price"])

                    pe_mid = put_side * put_mid_t
                    ce_mid = call_side * call_mid_t

                    pnl_mtm = (pe_mid + ce_mid) * lot_size * contracts_open - net_entry

                    delta_pc_t, gamma_pc_t, vega_pc_t, theta_pc_t = (
                        self._compute_greeks_per_contract(
                            pt, ct, put_side, call_side, lot_size
                        )
                    )
                    delta = delta_pc_t * contracts_open
                    gamma = gamma_pc_t * contracts_open
                    vega = vega_pc_t * contracts_open
                    theta = theta_pc_t * contracts_open

                # update S / smoothed_iv
                S_curr = last_rec["S"]
                if curr_date in options.index:
                    S_curr = options.loc[curr_date, "spot_price"].iloc[0]

                iv_curr = last_rec["iv"]
                if (
                    "smoothed_iv" in today_chain.columns
                    and not put_today.empty
                    and not call_today.empty
                ):
                    iv_curr = 0.5 * (
                        put_today.iloc[0]["smoothed_iv"]
                        + call_today.iloc[0]["smoothed_iv"]
                    )

                hedge_pnl = 0.0  # no hedge baseline
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
                        expiry_date=expiry_date,
                        put_quote=pt,
                        call_quote=ct,
                        put_entry=float(put_entry),
                        call_entry=float(call_entry),
                        lot_size=lot_size,
                        spot=float(S_curr),
                        volatility=float(iv_curr),
                    )
                    if margin_pc_curr is not None:
                        latest_margin_pc = margin_pc_curr

                initial_margin_req = (latest_margin_pc or 0.0) * contracts_open
                financing_pnl = 0.0
                maintenance_margin_req = 0.0
                margin_excess = np.nan
                margin_deficit = np.nan
                in_margin_call = False
                margin_call_days = 0
                forced_liquidation = False
                contracts_liquidated = 0
                margin_status = None
                if margin_account is not None:
                    margin_status = margin_account.evaluate(
                        equity=equity_running + delta_pnl_market,
                        initial_margin_requirement=initial_margin_req,
                        open_contracts=contracts_open,
                    )
                    financing_pnl = margin_status.financing_pnl
                    maintenance_margin_req = (
                        margin_status.maintenance_margin_requirement
                    )
                    margin_excess = margin_status.margin_excess
                    margin_deficit = margin_status.margin_deficit
                    in_margin_call = margin_status.in_margin_call
                    margin_call_days = margin_status.margin_call_days
                    forced_liquidation = margin_status.forced_liquidation
                    contracts_liquidated = margin_status.contracts_to_liquidate

                delta_pnl = delta_pnl_market + financing_pnl

                mtm_records.append(
                    {
                        "date": curr_date,
                        "S": S_curr,
                        "iv": iv_curr,
                        "delta_pnl": delta_pnl,
                        "delta": delta,
                        "net_delta": net_delta,
                        "gamma": gamma,
                        "vega": vega,
                        "theta": theta,
                        "hedge_qty": hedge_qty,
                        "hedge_price_prev": hedge_price_entry,
                        "hedge_pnl": hedge_pnl,
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
                        "financing_pnl": financing_pnl,
                    }
                )
                equity_running += delta_pnl

                if (
                    margin_status is not None
                    and margin_status.forced_liquidation
                    and margin_status.contracts_to_liquidate > 0
                    and pt is not None
                    and ct is not None
                ):
                    put_exit = pt["ask_price"] + cfg.slip_ask
                    call_exit = ct["ask_price"] + cfg.slip_ask
                    contracts_to_close = margin_status.contracts_to_liquidate
                    contracts_after = contracts_open - contracts_to_close
                    pnl_per_contract = (
                        put_side * (put_exit - put_entry)
                        + call_side * (call_exit - call_entry)
                    ) * lot_size
                    real_pnl_closed = pnl_per_contract * contracts_to_close + hedge_pnl
                    pnl_net_closed = real_pnl_closed - (
                        roundtrip_comm_pc * contracts_to_close
                    )

                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": curr_date,
                            "entry_dte": chosen_dte,
                            "expiry_date": expiry_date,
                            "contracts": contracts_to_close,
                            "put_strike": put_q["strike"],
                            "call_strike": call_q["strike"],
                            "put_entry": put_entry,
                            "call_entry": call_entry,
                            "put_exit": put_exit,
                            "call_exit": call_exit,
                            "pnl": pnl_net_closed,
                            "risk_per_contract": risk_pc,
                            "risk_worst_scenario": risk_scenario,
                            "margin_per_contract": latest_margin_pc,
                            "exit_type": (
                                "Margin Call Liquidation"
                                if contracts_after == 0
                                else "Margin Call Partial Liquidation"
                            ),
                        }
                    )

                    if contracts_after == 0:
                        forced_delta_pnl = (
                            pnl_net_closed - prev_mtm_before
                        ) + financing_pnl
                        equity_running += (
                            forced_delta_pnl - mtm_records[-1]["delta_pnl"]
                        )
                        mtm_records[-1].update(
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
                                "margin_excess": equity_running,
                                "margin_deficit": 0.0,
                                "in_margin_call": False,
                                "forced_liquidation": True,
                                "contracts_liquidated": contracts_to_close,
                            }
                        )
                        exited = True
                        break

                    ratio_remaining = contracts_after / contracts_open
                    pnl_mtm_remaining = pnl_mtm * ratio_remaining
                    forced_delta_pnl = (
                        pnl_net_closed + pnl_mtm_remaining - prev_mtm_before
                    ) + financing_pnl
                    equity_running += forced_delta_pnl - mtm_records[-1]["delta_pnl"]
                    mtm_records[-1].update(
                        {
                            "delta_pnl": forced_delta_pnl,
                            "delta": delta * ratio_remaining,
                            "net_delta": net_delta * ratio_remaining,
                            "gamma": gamma * ratio_remaining,
                            "vega": vega * ratio_remaining,
                            "theta": theta * ratio_remaining,
                            "open_contracts": contracts_after,
                            "initial_margin_requirement": (
                                (latest_margin_pc or 0.0) * contracts_after
                            ),
                            "maintenance_margin_requirement": (
                                (latest_margin_pc or 0.0)
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
                    contracts_open = contracts_after
                    net_entry *= ratio_remaining
                    prev_mtm = pnl_mtm_remaining
                    continue

                # realized P&L (closing both legs with slippage)
                if put_today.empty or call_today.empty:
                    prev_mtm = pnl_mtm
                    continue

                pt_exit = put_today.iloc[0]
                ct_exit = call_today.iloc[0]

                put_exit = pt_exit["ask_price"] + cfg.slip_ask  # buy back
                call_exit = ct_exit["ask_price"] + cfg.slip_ask

                pnl_per_contract = (
                    put_side * (put_exit - put_entry)
                    + call_side * (call_exit - call_entry)
                ) * lot_size
                real_pnl = pnl_per_contract * contracts_open + hedge_pnl

                exit_type = None
                if curr_date >= hold_date:
                    exit_type = "Holding Period"

                if exit_type is None:
                    prev_mtm = pnl_mtm
                    continue

                pnl_net = real_pnl - (roundtrip_comm_pc * contracts_open)
                exit_delta_pnl = (pnl_net - prev_mtm_before) + financing_pnl
                equity_running += exit_delta_pnl - mtm_records[-1]["delta_pnl"]

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": curr_date,
                        "entry_dte": chosen_dte,
                        "expiry_date": expiry_date,
                        "contracts": contracts_open,
                        "put_strike": put_q["strike"],
                        "call_strike": call_q["strike"],
                        "put_entry": put_entry,
                        "call_entry": call_entry,
                        "put_exit": put_exit,
                        "call_exit": call_exit,
                        "pnl": pnl_net,
                        "risk_per_contract": risk_pc,
                        "risk_worst_scenario": risk_scenario,
                        "margin_per_contract": latest_margin_pc,
                        "exit_type": exit_type,
                    }
                )

                # overwrite last MTM with realized pnl and flat Greeks
                mtm_records[-1].update(
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
                        "margin_excess": equity_running,
                        "margin_deficit": 0.0,
                        "in_margin_call": False,
                    }
                )

                exited = True
                break

            if not exited:
                # Keep the open-position MTM path and stop to avoid overlap.
                break

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

        # equity curve initialized with provided capital
        mtm["equity"] = capital + mtm["delta_pnl"].cumsum()

        return pd.DataFrame(trades), mtm
