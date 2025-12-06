import numpy as np
import pandas as pd

from volatility_trading.signals import Signal
from volatility_trading.filters import Filter
from volatility_trading.backtesting import BacktestConfig, SliceContext

from ..base_strategy import Strategy

from volatility_trading.options.greeks import bs_greeks


class VRPHarvestingStrategy2(Strategy):
    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        target_dte: int = 30,
        rebalance_days: int = 5,
    ):
        """
        Baseline VRP harvesting strategy:
        - Short ATM straddle when signal is ON.
        - Target ~30D maturity (by dte).
        - Positions are rolled / rebalanced every `rebalance_days`
          into a fresh ~30D straddle (when signal remains ON).
        - No delta-hedge in the baseline.
        """
        super().__init__(signal=signal, filters=filters)
        self.target_dte = target_dte
        self.rebalance_days = rebalance_days

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

    # --------- Main entry point ---------

    def run(self, ctx: SliceContext):
        data = ctx.data
        capital = ctx.capital
        cfg = ctx.config

        options = data["options"]
        features = data.get("features")  # expected: DataFrame with iv_atm etc.
        hedge = data.get("hedge")        # not used yet, but kept in signature

        # For now: always-on short vol (filters will switch us OFF if needed)
        series = pd.Series(0, index=options.index)
        signals = self.signal.generate_signals(series)

        for f in self.filters:
            signals = f.apply(signals, {"features": features})

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
        signals: pd.DataFrame,
        features: pd.DataFrame | None,
        hedge: pd.Series | None,
        capital: float,
        cfg: BacktestConfig,
    ):
        """
        Baseline backtest:
        - Short 1 ATM straddle when signal is ON.
        - Close and realize P&L when:
          - `rebalance_days` have passed (roll into a new 30D),
          - (No SL/TP yet in this baseline).
        """

        lot_size = cfg.lot_size
        roundtrip_comm = 2 * cfg.commission_per_leg

        signals = signals.sort_index()
        options = options.sort_index()

        trades: list[dict] = []
        mtm_records: list[dict] = []
        last_exit = None

        # We use 1 contract for now (no capital-based sizing)
        contracts = 1

        for entry_date, row in signals.iterrows():
            # Only act when short signal is ON
            if not row["short"]:
                continue
            if entry_date not in options.index:
                continue
            if last_exit is not None and entry_date <= last_exit:
                # don't overlap trades in this simple baseline
                continue

            chain = options.loc[entry_date]

            # --- choose expiry closest to target DTE if available ---
            if "dte" in chain.columns:
                dtes = chain["dte"].dropna().unique()
                if len(dtes) == 0:
                    continue
                chosen_dte = min(dtes, key=lambda d: abs(d - self.target_dte))
                chain = chain[chain["dte"] == chosen_dte]
                print("Chosen DTE:", chosen_dte)

            if "expiry" not in chain.columns or chain.empty:
                continue

            expiry = chain["expiry"].iloc[0]

            # --- pick ATM-ish put and call using delta ~ +/- 0.5 ---
            puts = chain[chain["option_type"] == "P"]
            calls = chain[chain["option_type"] == "C"]

            put_q = self._pick_quote(puts, tgt=-0.5, delta_tolerance=0.05)
            call_q = self._pick_quote(calls, tgt=+0.5, delta_tolerance=0.05)
            if put_q is None or call_q is None:
                continue

            # short straddle: short put & short call
            put_side = -1
            call_side = -1

            # entry prices: conservative (sell at bid - slip)
            put_bid = put_q["bid"]
            call_bid = call_q["bid"]
            put_entry = put_bid - cfg.slip_bid
            call_entry = call_bid - cfg.slip_bid



            # Greeks per contract
            delta_pc, gamma_pc, vega_pc, theta_pc = self._compute_greeks_per_contract(
                put_q, call_q, put_side, call_side, lot_size
            )

            net_entry = (put_side * put_entry + call_side * call_entry) * lot_size * contracts

            S_entry = chain["underlying_last"].iloc[0]
            # Best: use IV from the actual legs if you have it
            if "iv" in put_q.index and "iv" in call_q.index:
                iv_entry = 0.5 * (put_q["iv"] + call_q["iv"])
            elif (
                features is not None
                and entry_date in features.index
                and "iv_atm" in features.columns
            ):
                iv_entry = features.loc[entry_date, "iv_atm"]
            else:
                iv_entry = np.nan

            # initial Greeks across all contracts
            delta = delta_pc * contracts
            gamma = gamma_pc * contracts
            vega = vega_pc * contracts
            theta = theta_pc * contracts

            # no hedge in baseline
            hedge_qty = 0
            hedge_price_entry = np.nan
            net_delta = delta

            entry_idx = len(mtm_records)
            mtm_records.append(
                {
                    "date": entry_date,
                    "S": S_entry,
                    "iv": iv_entry,
                    "delta_pnl": -roundtrip_comm,  # pay comm at entry
                    "delta": delta,
                    "net_delta": net_delta,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "hedge_qty": hedge_qty,
                    "hedge_price_prev": hedge_price_entry,
                    "hedge_pnl": 0.0,
                }
            )
            prev_mtm = 0.0  # value-based MTM; delta_pnl tracks realized increments

            # rebalance/roll date: after `rebalance_days`
            rebalance_date = entry_date + pd.Timedelta(days=self.rebalance_days)
            future_dates = sorted(options.index[options.index > entry_date].unique())

            exited = False

            for curr_date in future_dates:
                today_chain = options.loc[curr_date]
                today_chain = today_chain[today_chain["expiry"] == expiry]

                put_today = today_chain[
                    (today_chain["option_type"] == "P")
                    & (today_chain["strike"] == put_q["strike"])
                ]
                call_today = today_chain[
                    (today_chain["option_type"] == "C")
                    & (today_chain["strike"] == call_q["strike"])
                ]

                last_rec = mtm_records[-1]

                if put_today.empty or call_today.empty:
                    # carry last Greeks; no MTM update from options
                    pnl_mtm = prev_mtm
                    delta = last_rec["delta"]
                    gamma = last_rec["gamma"]
                    vega = last_rec["vega"]
                    theta = last_rec["theta"]
                    S_curr = last_rec["S"]
                    iv_curr = last_rec["iv"]
                else:
                    # recompute MTM
                    pt = put_today.iloc[0]
                    ct = call_today.iloc[0]

                    put_mid_t = 0.5 * (pt["bid"] + pt["ask"])
                    call_mid_t = 0.5 * (ct["bid"] + ct["ask"])

                    pe_mid = put_side * put_mid_t
                    ce_mid = call_side * call_mid_t

                    pnl_mtm = (pe_mid + ce_mid) * lot_size * contracts - net_entry

                    delta_pc_t, gamma_pc_t, vega_pc_t, theta_pc_t = (
                        self._compute_greeks_per_contract(
                            pt, ct, put_side, call_side, lot_size
                        )
                    )
                    delta = delta_pc_t * contracts
                    gamma = gamma_pc_t * contracts
                    vega = vega_pc_t * contracts
                    theta = theta_pc_t * contracts

                    # update S / iv
                    S_curr = last_rec["S"]
                    if curr_date in options.index:
                        S_curr = options.loc[curr_date, "underlying_last"].iloc[0]

                    iv_curr = last_rec["iv"]
                    if (
                        "iv" in today_chain.columns
                        and not put_today.empty
                        and not call_today.empty
                    ):
                        iv_curr = 0.5 * (
                            put_today.iloc[0]["iv"] + call_today.iloc[0]["iv"]
                        )
                    elif (
                        features is not None
                        and curr_date in features.index
                        and "iv_atm" in features.columns
                    ):
                        iv_curr = features.loc[curr_date, "iv_atm"]

                hedge_pnl = 0.0  # no hedge baseline
                net_delta = delta

                delta_pnl = (pnl_mtm - prev_mtm) + hedge_pnl

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
                    }
                )
                prev_mtm = pnl_mtm

                # realized P&L (closing both legs with slippage)
                if put_today.empty or call_today.empty:
                    continue

                pt_exit = put_today.iloc[0]
                ct_exit = call_today.iloc[0]

                put_exit = pt_exit["ask"] + cfg.slip_ask  # buy back
                call_exit = ct_exit["ask"] + cfg.slip_ask

                pnl_per_contract = (
                    put_side * (put_exit - put_entry)
                    + call_side * (call_exit - call_entry)
                ) * lot_size
                real_pnl = pnl_per_contract * contracts + hedge_pnl

                exit_type = None
                if curr_date >= rebalance_date:
                    exit_type = "Rebalance"

                if exit_type is None:
                    continue

                pnl_net = real_pnl - roundtrip_comm

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": curr_date,
                        "expiry": expiry,
                        "contracts": contracts,
                        "put_strike": put_q["strike"],
                        "call_strike": call_q["strike"],
                        "put_entry": put_entry,
                        "call_entry": call_entry,
                        "put_exit": put_exit,
                        "call_exit": call_exit,
                        "pnl": pnl_net,
                        "exit_type": exit_type,
                    }
                )

                # overwrite last MTM with realized pnl and flat Greeks
                mtm_records[-1].update(
                    {
                        "delta_pnl": pnl_net,
                        "delta": 0.0,
                        "net_delta": 0.0,
                        "gamma": 0.0,
                        "vega": 0.0,
                        "theta": 0.0,
                        "hedge_qty": 0.0,
                    }
                )

                last_exit = curr_date
                exited = True
                break

            if not exited:
                # drop MTM records for this aborted trade
                del mtm_records[entry_idx:]

        if not mtm_records:
            return pd.DataFrame(trades), pd.DataFrame()

        mtm_agg = pd.DataFrame(mtm_records).set_index("date").sort_index()
        mtm = (
            mtm_agg.groupby("date")
            .agg(
                {
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
            )
        )

        # equity curve initialized with provided capital
        mtm["equity"] = capital + mtm["delta_pnl"].cumsum()

        return pd.DataFrame(trades), mtm
    


class VRPHarvestingStrategy(Strategy):
    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        holding_period: int = 5,
        target_dte: int = 30,
        max_dte_diff: int = 7

    ):
        """
        Baseline VRP harvesting strategy:
        - short ATM straddle when signal is ON
        - 30D target maturity (by dte)
        - simple SL/TP in units of a stress-based risk per contract
        - no delta-hedge (for now)
        """
        super().__init__(signal=signal, filters=filters)
        self.holding_period = holding_period
        self.target_dte = target_dte
        self.max_dte_diff = max_dte_diff

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
    def choose_vrp_expiry(
        chain: pd.DataFrame,
        target_dte: int = 30,
        max_dte_diff: int = 7,
        min_atm_quotes: int = 2,
    ) -> int | None:
        """
        Pick a single expiry for the VRP straddle:
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
            # assumes you have 'S' or 'underlying_last'
            S = sub["underlying_last"].iloc[0]
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
        hedge = data.get("hedge")        # not used yet, but kept in signature

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
        - Short 1 ATM straddle when signal is ON.
        - No hedging.
        - SL / TP / holding-period exits.
        """

        lot_size = cfg.lot_size
        roundtrip_comm = 2 * cfg.commission_per_leg

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
        block_until = None

        # We use 1 contract for now (no capital-based sizing)
        contracts = 1

        for entry_date, row in sig_df.iterrows():
            if not row["on"]:
                continue
            if entry_date not in options.index:
                continue
            if block_until is not None and entry_date < block_until:
                continue

            chain = options.loc[entry_date]

            # --- choose expiry closest to target DTE if available ---
            chain = options.loc[entry_date]
            chosen_dte = self.choose_vrp_expiry(
                chain=chain, 
                target_dte=self.target_dte, 
                max_dte_diff=self.max_dte_diff
            )
            if chosen_dte is None:
                continue

            chain = chain[chain["dte"] == chosen_dte]
            expiry = chain["expiry"].iloc[0]

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

            put_bid = put_q["bid"]
            call_bid = call_q["bid"]
            put_entry = put_bid - cfg.slip_bid
            call_entry = call_bid - cfg.slip_bid

            # Greeks per contract
            delta_pc, gamma_pc, vega_pc, theta_pc = self._compute_greeks_per_contract(
                put_q, call_q, put_side, call_side, lot_size
            )

            S_entry = chain["underlying_last"].iloc[0]
            # Best: use IV from the actual legs if you have it
            if "iv" in put_q.index and "iv" in call_q.index:
                iv_entry = 0.5 * (put_q["iv"] + call_q["iv"])
            elif (
                features is not None
                and entry_date in features.index
                and "iv_atm" in features.columns
            ):
                iv_entry = features.loc[entry_date, "iv_atm"]
            else:
                iv_entry = np.nan

            # net entry value with sign convention (short)
            net_entry = (put_side * put_entry + call_side * call_entry) * lot_size * contracts

            # initial Greeks across all contracts
            delta = delta_pc * contracts
            gamma = gamma_pc * contracts
            vega = vega_pc * contracts
            theta = theta_pc * contracts

            # no hedge in baseline
            hedge_qty = 0
            hedge_price_entry = np.nan
            net_delta = delta

            entry_idx = len(mtm_records)
            mtm_records.append(
                {
                    "date": entry_date,
                    "S": S_entry,
                    "iv": iv_entry,
                    "delta_pnl": -roundtrip_comm,  # pay comm at entry
                    "delta": delta,
                    "net_delta": net_delta,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "hedge_qty": hedge_qty,
                    "hedge_price_prev": hedge_price_entry,
                    "hedge_pnl": 0.0,
                }
            )
            prev_mtm = 0.0  # value-based MTM; delta_pnl tracks realized increments

            hold_date = entry_date + pd.Timedelta(days=self.holding_period)
            block_until = hold_date
            future_dates = sorted(options.index[options.index > entry_date].unique())
            exited = False

            for curr_date in future_dates:
                today_chain = options.loc[curr_date]
                today_chain = today_chain[today_chain["expiry"] == expiry]

                put_today = today_chain[
                    (today_chain["option_type"] == "P")
                    & (today_chain["strike"] == put_q["strike"])
                ]
                call_today = today_chain[
                    (today_chain["option_type"] == "C")
                    & (today_chain["strike"] == call_q["strike"])
                ]

                last_rec = mtm_records[-1]

                if put_today.empty or call_today.empty:
                    # carry last Greeks; no MTM update from options
                    pnl_mtm = prev_mtm
                    delta = last_rec["delta"]
                    gamma = last_rec["gamma"]
                    vega = last_rec["vega"]
                    theta = last_rec["theta"]
                else:
                    # recompute MTM
                    pt = put_today.iloc[0]
                    ct = call_today.iloc[0]

                    put_mid_t = 0.5 * (pt["bid"] + pt["ask"])
                    call_mid_t = 0.5 * (ct["bid"] + ct["ask"])

                    pe_mid = put_side * put_mid_t
                    ce_mid = call_side * call_mid_t

                    pnl_mtm = (pe_mid + ce_mid) * lot_size * contracts - net_entry

                    delta_pc_t, gamma_pc_t, vega_pc_t, theta_pc_t = self._compute_greeks_per_contract(
                        pt, ct, put_side, call_side, lot_size
                    )
                    delta = delta_pc_t * contracts
                    gamma = gamma_pc_t * contracts
                    vega = vega_pc_t * contracts
                    theta = theta_pc_t * contracts

                # update S / iv
                S_curr = last_rec["S"]
                if curr_date in options.index:
                    S_curr = options.loc[curr_date, "underlying_last"].iloc[0]

                iv_curr = last_rec["iv"]
                if (
                    "iv" in today_chain.columns
                    and not put_today.empty
                    and not call_today.empty
                ):
                    iv_curr = 0.5 * (
                        put_today.iloc[0]["iv"] + call_today.iloc[0]["iv"]
                    )

                hedge_pnl = 0.0  # no hedge baseline
                net_delta = delta

                delta_pnl = (pnl_mtm - prev_mtm) + hedge_pnl

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
                    }
                )
                prev_mtm = pnl_mtm

                # realized P&L (closing both legs with slippage)
                if put_today.empty or call_today.empty:
                    continue

                pt_exit = put_today.iloc[0]
                ct_exit = call_today.iloc[0]

                put_exit = pt_exit["ask"] + cfg.slip_ask  # buy back
                call_exit = ct_exit["ask"] + cfg.slip_ask

                pnl_per_contract = (
                    put_side * (put_exit - put_entry)
                    + call_side * (call_exit - call_entry)
                ) * lot_size
                real_pnl = pnl_per_contract * contracts + hedge_pnl

                exit_type = None
                if curr_date >= hold_date:
                    exit_type = "Holding Period"

                if exit_type is None:
                    continue

                pnl_net = real_pnl - roundtrip_comm

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": curr_date,
                        "entry_dte": chosen_dte,
                        "expiry": expiry,
                        "contracts": contracts,
                        "put_strike": put_q["strike"],
                        "call_strike": call_q["strike"],
                        "put_entry": put_entry,
                        "call_entry": call_entry,
                        "put_exit": put_exit,
                        "call_exit": call_exit,
                        "pnl": pnl_net,
                        "exit_type": exit_type,
                    }
                )

                # overwrite last MTM with realized pnl and flat Greeks
                mtm_records[-1].update(
                    {
                        "delta_pnl": pnl_net,
                        "delta": 0.0,
                        "net_delta": 0.0,
                        "gamma": 0.0,
                        "vega": 0.0,
                        "theta": 0.0,
                        "hedge_qty": 0.0,
                    }
                )

                last_exit = curr_date
                exited = True
                break

            if not exited:
                # drop MTM records for this aborted trade
                del mtm_records[entry_idx:]

        if not mtm_records:
            return pd.DataFrame(trades), pd.DataFrame()

        mtm_agg = pd.DataFrame(mtm_records).set_index("date").sort_index()
        mtm = (
            mtm_agg.groupby("date")
            .agg(
                {
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
            )
        )

        # equity curve initialized with provided capital
        mtm["equity"] = capital + mtm["delta_pnl"].cumsum()

        return pd.DataFrame(trades), mtm