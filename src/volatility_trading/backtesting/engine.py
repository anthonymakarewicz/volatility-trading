import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from typing import Mapping, Any
from dataclasses import dataclass


# -------------
# Prototype
# -------------

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    leverage: float = 1.0
    # maybe later: slippage, commission, margin model, etc.

@dataclass
class SliceContext:
    #data: dict[str, pd.DataFrame]
    data: Mapping[str, Any]   # {"options": df, "features": df, "hedge": series, ...}
    params: dict
    config: BacktestConfig
    capital: float # current capital


class OptionStrategy(ABC):
    pass

class Straddle(OptionStrategy):
    def __init__(K, T, direction="long"):
        pass
        
    def setup_butterfly(futures_price, options_chain, direction='long'):
        butterfly = pd.DataFrame()

        butterfly['Option Type'] = ['CE', 'PE']
        atm_strike_price = 50 * (round(futures_price / 50))
        butterfly['Strike Price'] = atm_strike_price
        butterfly['position'] = 1

        butterfly['premium'] = butterfly.apply(
            lambda r: get_premium(r, options_chain), axis=1)

        deviation = round(butterfly.premium.sum()/50) * 50
        butterfly.loc['2'] = ['CE', atm_strike_price+deviation, -1, np.nan]
        butterfly.loc['3'] = ['PE', atm_strike_price-deviation, -1, np.nan]

        if direction == 'long':
            butterfly['position'] *= -1

        butterfly['premium'] = butterfly.apply(
            lambda r: get_premium(r, options_chain), axis=1)

        return butterfly




class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, config:BacktestConfig):
        """
        data: dict of named DataFrames/Series, e.g.
            {
                "options": options_df,      # Option chain to execute trades
                "features": features_df,   # For signals and filters
                "hedge": hedge_series,    # for delta hedging
            }
        """
        self.data = data
        self.strategy = strategy
        self.config = config

    def _rolling_windows(self):
        # your existing logic
        ...

    def _slice_data(self, start, end):
        sliced = {}
        for name, df in self.data.items():
            sliced[name] = df.loc[start:end]
        return sliced

    def run(self):
        results = []
        for window in self._rolling_windows():
            # Do the optmiziation 
            # best_params

            data_slice = self._slice_data(window.test_start, window.test_end)

            ctx = SliceContext(
                data=data_slice,
                params=best_params,
                config=self.config,
                capital=current_capital
            )

            trades, equity = self.strategy.run_slice(ctx)
            results.append((trades, equity))
        # aggregate results, compute metrics, etc.
        return results
    

class Strategy(ABC):
    def __init__(self, signal: Signal, filters: list[Filter] | None = None):
        self.signal = signal
        self.filters = filters or []

    def _set_all_params(self, params):
        """Utility to split and set strategy & filter params"""
        s = {
            k.split("__",1)[1]:v 
            for k,v in params.items() 
            if k.startswith("signal__")
        }
        self.signal.set_params(**s)

        for filt in self.filters:
            name = filt.__class__.__name__.lower()+"__"
            f = {
                k.split("__",1)[1]:v 
                for k,v in params.items() 
                if k.startswith(name)
            }
            filt.set_params(**f)
    
class VRPStrategy(Strategy):
    def __init__(self, signal: Signal, filters: list[Filter] | None = None):
        self.signal = signal
        self.filters = filters or []

    def run_slice(self, ctx):
        data = ctx["data"]
        capital = ctx["capital"]
        params = ctx["params"]

        options  = data["options"]      
        features = data["features"]     
        hedge    = data.get("hedge")  

        # build signals and apply filters
        series = features["iv_atm"]  
        signals = self.signal.generate_signals(series)
        for f in self.filters:
            signals = f.apply(signals, ctx={"features": features})

        # 4) use signals + options data to simulate trades / P&L
        trades, mtm = self._simulate_short_straddles(
            options=options,
            signals=signals,
            features=features,
            capital=capital,
            hedge=hedge,
            params=params,
        )

        return trades, mtm
    


# -------------
# Current class
# -------------

class Backtester:
    def __init__(
        self,
        options, 
        synthetic_skew, 
        vix,
        hedge_series,
        strategy,
        filters,
        position_sizing,
        find_viable_dtes,
        params_grid=None,
        train_window=252,
        val_window=63,
        test_window=63,
        step=None,
        target_dte=30, 
        target_delta_otm=0.25, delta_tolerance=0.05, 
        holding_period=5,
        stop_loss_pct=0.8, take_profit_pct=0.5,
        theta_decay_weekend=200,
        lot_size=100,
        hedge_size=50,
        risk_pc_floor=750,
        slip_ask=0.01, slip_bid=0.01,
        commission_per_leg=1.00, 
        initial_capital=100000
    ):
        # data
        self.options = options
        self.synthetic_skew = synthetic_skew
        self.vix = vix
        self.hedge_series = hedge_series

        # strategy object & filter list objects
        self.strategy = strategy
        self.filters = filters

        # external functions
        self.position_sizing = position_sizing
        self.find_viable_dtes = find_viable_dtes

        # parameters for tuning
        self.params_grid = params_grid

        # parameters for walk froward
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step = step
        if self.step is None:
            self.step = self.test_window

        # risk reversal params
        self.target_dte = target_dte
        self.target_delta_otm = target_delta_otm
        self.delta_tolerance = delta_tolerance

        # risk management params
        self.holding_period = holding_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.theta_decay_weekend = theta_decay_weekend

        # backtest params
        self.lot_size = lot_size
        self.hedge_size = hedge_size
        self.risk_pc_floor = risk_pc_floor
        self.slip_ask = slip_ask
        self.slip_bid = slip_bid
        self.commission_per_leg = commission_per_leg
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # --- auto-compute the warm lookback 
        #     for computing signals and filters ---
        base_windows = [strategy.window]
        base_windows += [
            f.window for f in filters if hasattr(f, 'window')
        ]
        if self.params_grid is not None:
            for key, values in self.params_grid.items():
                if key.endswith("__window"):
                    for w in values:
                        base_windows.append(int(w))

        # the lookback window is the maximum window size
        self.lookback = max(base_windows)


    ##################
    # Static methods #
    ##################

    # --- Public Static Methods ---
    @staticmethod
    def compute_sharpe_ratio(returns, rf=0.0):
        return (
            (returns.mean() - rf)/returns.std()*np.sqrt(252) 
            if not returns.empty 
            else -np.inf
        )

    @staticmethod
    def stress_test(mtm_daily, scenarios):
        pnl_df = pd.DataFrame(index=mtm_daily.index)
        for name, shock in scenarios.items():
            dS = shock['dS_pct'] * mtm_daily['S']
            pnl_df[f'PnL_{name}'] = (
                mtm_daily['net_delta_prev'] * dS
                + 0.5 * mtm_daily['gamma_prev'] * dS**2
                +       mtm_daily['vega_prev']  * shock['dSigma']
                +       mtm_daily['theta_prev'] * shock['dt']
            )
        return pnl_df
    
    #  --- Private Static Methods ---
    @staticmethod
    def _pick_quote(df_leg, tgt, delta_tolerance=0.05):
        df2 = df_leg.copy()
        df2['d_err'] = (df2['delta'] - tgt).abs()
        df2 = df2[df2['d_err'] <= delta_tolerance]
        if df2.empty:
            return None
        return df2.iloc[df2['d_err'].values.argmin()]


    ###################
    # Private methods #
    ###################
    
    def _apply_filters(self, start_date, end_date, raw_signals):
        ctx = self._make_context(start_date, end_date)
        signals = raw_signals.copy()
        for filt in self.filters:
            signals = filt.apply(signals, ctx)
        return signals
    
    def _make_context(self, start_date, end_date):
        """Stores relevent info to pass to strategy and filters"""
        idx = self.synthetic_skew.index
        pos_start = idx.get_loc(start_date)
        pos_end = idx.get_loc(end_date)
        start = idx[
            max(0, pos_end - (self.lookback + (pos_end - pos_start)))
        ]
        
        skew = self.synthetic_skew.loc[start:end_date, 'skew_norm']
        iv_atm = self.synthetic_skew.loc[start:end_date, 'iv_atm_30']
        vix = self.vix.loc[start:end_date]
        return {
            'skew_norm': skew, 'iv_atm': iv_atm, 'vix': vix
        }

    def _generate_signals(self, start_date, end_date):
        idx = self.synthetic_skew.index
        pos_start = idx.get_loc(start_date)
        pos_end = idx.get_loc(end_date)
        start = idx[max(0, pos_end - (self.lookback + (pos_end - pos_start)))]

        skew = self.synthetic_skew.loc[start:end_date, 'skew_norm']
        signals = self.strategy.generate_signals(skew)
        return signals

    def _compute_score(self, trades, eq, w_sr, w_max_dd, w_pf, pf_eps=1e-6):
        if trades.empty or eq.empty:
            return -np.inf
        
        # Sharpe ratio 
        daily_ret = eq.equity.pct_change().dropna()
        sr    = Backtester.compute_sharpe_ratio(daily_ret)

        # Max drawdown
        peak   = eq.equity.cummax()
        dd     = (eq.equity - peak) / peak
        max_dd = dd.min()

        # Profit factor with floor & dynamic cap
        wins    = trades.loc[trades.pnl > 0, "pnl"].sum()
        losses  = -trades.loc[trades.pnl <= 0, "pnl"].sum()
        losses  = max(losses, pf_eps)         # avoid division by zero
        raw_pf  = wins / losses

        n_trades = len(trades)
        if losses <= pf_eps:
            # zero losses, apply piecewise cap
            if n_trades <= 3:
                pf_cap = 1.5
            elif n_trades <= 6:
                pf_cap = 2.0
            elif n_trades <= 15:
                pf_cap = 3.0
            else:
                pf_cap = 5.0
            pf = min(raw_pf, pf_cap)
        else:
            # at least one loss, trust raw PF
            pf = raw_pf

        # Composite score
        return w_sr*sr - w_max_dd*abs(max_dd) + w_pf*pf
    
    def _score_params(
            self, train_idx, val_idx,
            top_k=5, w_sr=0.5, w_max_dd=0.3, w_pf=0.2, 
    ):
        """
        For each hyper–parameter combo:
          1. Backtest on the training slice and score via our composite 
          metric (Sharpe, maximum drawdown, profit factor).
          2. Keep the top_k.
          3. Re–test those on the validation slice, again with
             the same composite metric, and pick the best overall.
        """
        opts_train = self.options.loc[train_idx]
        opts_val = self.options.loc[val_idx]
        first_train, last_train = train_idx[0], train_idx[-1]
        first_val, last_val = val_idx[0], val_idx[-1]
        grid = list(ParameterGrid(self.params_grid))

        def score_on_slice(params, opts, idx, first, last):
            """Apply params, generate signals, backtest, compute score."""
            self._set_all_params(params)
            sig = self._generate_signals(first, last)
            sig = self._apply_filters(first, last, sig)
            trades, eq = self._backtest_risk_reversal(
                opts, sig.loc[idx], capital=self.initial_capital
            )
            return (
                self._compute_score(trades, eq, w_sr, w_max_dd, w_pf), params
            )

        # 1) Score training slice in parallel
        train_results = Parallel(n_jobs=-1)(
            delayed(score_on_slice)(
                p, opts_train, train_idx, first_train, last_train
            )
            for p in grid
        )
        
        if not train_results:
            return None

        # 2) Pick top_k by training score
        best_candidates = [
            params 
            for _, params in sorted(
                train_results,
                key=lambda x: x[0], # sort by the score only
                reverse=True
            )[:top_k]
        ]

        # 3) Score those on validation slice (serially)
        best_score, best_params = -np.inf, None
        for params in best_candidates:
            score_val, _ = score_on_slice(
                params, opts_val, val_idx, first_val, last_val
            )
            if score_val > best_score:
                best_score, best_params = score_val, params

        return best_params

    def _set_all_params(self, params):
        """Utility to split and set strategy & filter params"""
        s = {
            k.split("__",1)[1]:v 
            for k,v in params.items() 
            if k.startswith("strategy__")
        }
        self.strategy.set_params(**s)

        for filt in self.filters:
            name = filt.__class__.__name__.lower()+"__"
            f = {
                k.split("__",1)[1]:v 
                for k,v in params.items() 
                if k.startswith(name)
            }
            filt.set_params(**f)

    def _compute_greeks_per_contract(self, put_q, call_q, put_side, call_side):
        delta = (
            put_side*put_q['delta'] + call_side*call_q['delta']
        ) * self.lot_size

        gamma = (
            put_side*put_q['gamma'] + call_side*call_q['gamma']
        ) * self.lot_size

        vega = (
            put_side*put_q['vega'] + call_side*call_q['vega']
        ) * self.lot_size

        theta = (
            put_side*put_q['theta'] + call_side*call_q['theta']
        ) * self.lot_size

        return delta, gamma, vega, theta

    def _backtest_risk_reversal(self, options, signals, capital):
        trades = []
        mtm_records = []  # daily MTM P&L and Greeks
        roundtrip_comm = 2 * self.commission_per_leg
        last_exit = None

        for entry_date, sig in signals.iterrows():
            if not (sig.get('long', False) or sig.get('short', False)):
                continue
            if entry_date not in options.index:
                continue
            if entry_date == last_exit:
                continue

            # --- choose expiry closest to target dte ---
            chain = options.loc[entry_date]
            d1, d2 = self.find_viable_dtes(
                chain, self.target_dte,
                target_delta_otm=self.target_delta_otm,
                delta_tolerance=self.delta_tolerance
            )
            dtes = [d for d in (d1, d2) if d is not None]
            if not dtes:
                continue
            chosen_dte = min(dtes, key=lambda d: abs(d - self.target_dte))
            chain = chain[chain['dte'] == chosen_dte]
            expiry = chain['expiry'].iloc[0]

            # --- pick entry quotes and strikes for put and call ---
            put_q = self._pick_quote(chain[chain.option_type=='P'], 
                                    -self.target_delta_otm, 
                                    self.delta_tolerance
            )
            call_q = self._pick_quote(chain[chain.option_type=='C'],  
                                     self.target_delta_otm, 
                                     self.delta_tolerance
            )
            if put_q is None or call_q is None:
                continue

            # --- decide put vs call side ---
            put_side  = -1 if sig['short'] else 1
            call_side =  1 if sig['short'] else -1

            # --- entry options prices per lot ---
            put_entry = (
                (put_q['bid'] - self.slip_bid) if sig['short']
                else (put_q['ask'] + self.slip_ask)
            )
            call_entry = (
                (call_q['ask'] + self.slip_ask) if sig['short']
                else (call_q['bid'] - self.slip_bid)
            )

            # --- compte risk per contract (pc)---
            # compute greeks per contract
            delta_pc, gamma_pc, vega_pc, theta_pc = (
                self._compute_greeks_per_contract(
                    put_q, call_q, put_side, call_side)
            )

            # how many futures to neutralize that delta
            hedge_qty_pc = int(round(-delta_pc / self.hedge_size))
            net_delta_pc = delta_pc + hedge_qty_pc * self.hedge_size

            # define stress-test shock parameters
            S_entry = options.loc[entry_date, 'underlying_last'].iloc[0]
            iv_entry = self.synthetic_skew.loc[entry_date, 'iv_atm_30']
            delta_S = 0.05 * S_entry          # 5% spot move
            delta_sigma_level = 0.03         # 3 vol-pts parallel shift
            delta_sigma_put_steep  = 0.04   # 4 vol-pts up on puts
            delta_sigma_call_steep = 0.02  # 2 vol-pts up on calls
            delta_sigma_flat = 0.02       # 2 vol-pt flat for long-skew
            dt = self.holding_period

            # delta, gamma & theta risk 
            delta_risk = abs(net_delta_pc) * delta_S
            gamma_risk = max(-0.5 * gamma_pc * (delta_S**2), 0.0) # only < 0
            theta_risk = max(-theta_pc * dt, 0.0) # only <0

            # level-shift vega risk (net exposure)
            vega_level_risk = abs(vega_pc) * delta_sigma_level

            # skew-shape vega risk (leg-by-leg)
            # leg-specific vegas
            vega_put  = put_side  * put_q['vega']  * self.lot_size
            vega_call = call_side * call_q['vega'] * self.lot_size

            # short-skew → skew-steepening (puts ↑, calls ↓)
            if sig['short']:
                vega_skew_risk = (
                    abs(vega_put)  * delta_sigma_put_steep +
                    abs(vega_call) * delta_sigma_call_steep
                )
            else:
                # long-skew → skew-flattening (puts ↓, calls ↑)
                vega_skew_risk = (
                    (abs(vega_put) + abs(vega_call)) * delta_sigma_flat
                )
            
            # final vega risk = worst of level-shift vs. skew-shape
            vega_risk = max(vega_level_risk, vega_skew_risk)

            # total worst-case risk per contract
            risk_pc = (
                delta_risk + gamma_risk + vega_risk + theta_risk
            )
            # prevent too small risk_pc to undervalue true risk
            risk_pc = max(risk_pc, self.risk_pc_floor)

            # --- position sizing ---
            z = self.strategy.get_z_score().loc[entry_date]
            contracts = self.position_sizing(
                z=z,
                capital=capital,
                risk_per_contract=risk_pc,
                entry_threshold=self.strategy.entry,
            )
            if contracts <= 0:
                continue

            # --- compute net entry and risk for all contracts ---
            net_entry = (
                (put_side*put_entry + call_side*call_entry) * 
                self.lot_size * contracts
            )
            total_risk = risk_pc * contracts

            # --- define stop loss and take profit levels ---
            sl_level = -self.stop_loss_pct * total_risk
            tp_level =  self.take_profit_pct * total_risk

            # --- compute Greeks across all contracts---
            delta = delta_pc * contracts
            gamma = gamma_pc * contracts
            vega  = vega_pc  * contracts
            theta = theta_pc * contracts

            # --- avoid very negative-theta trades during weekend  ---
            dow = entry_date.day_name()
            if dow == "Friday": 
                total_decay = 2 * theta
                if total_decay < -self.theta_decay_weekend: # only when paying decay
                    continue

            # --- delta hedge to nearest integer of contracts---
            hedge_qty = int(round(-delta/self.hedge_size))
            hedge_price_entry = self.hedge_series.loc[entry_date]
            net_delta = delta + (hedge_qty*self.hedge_size)

            # --- init MTM with entry comm, Greeks, and hedge info ---
            entry_idx = len(mtm_records)
            mtm_records.append({
                'date'             : entry_date,
                'S'                : S_entry,
                'iv'               : iv_entry,
                'delta_pnl'        : -roundtrip_comm, 
                'delta'            : delta,      # option-only Δ
                'net_delta'        : net_delta, # true post-hedge Δ
                'gamma'            : gamma,
                'vega'             : vega,
                'theta'            : theta,
                'hedge_qty'        : hedge_qty,
                'hedge_price_prev' : hedge_price_entry,
                'hedge_pnl'        : 0.0 # could include transac costs
            })
            prev_mtm = -roundtrip_comm

            # --- holding date and futures in-trade dates ---
            hold_date = (
                entry_date + pd.Timedelta(days=self.holding_period)
            )   
            future_dates = sorted(options.index[
                options.index > entry_date
            ].unique())

            exited = False
            for curr_date in future_dates:

                # -- same strikes and expiry used for entry ---
                today_chain = options.loc[curr_date]
                today_chain = today_chain[
                        today_chain['expiry'] == expiry
                ]
                put_today = today_chain[(
                    (today_chain.option_type=='P') &
                    (today_chain.strike == put_q['strike'])
                )]
                call_today = today_chain[(
                    (today_chain.option_type=='C') & 
                    (today_chain.strike == call_q['strike'])
                )]

                # --- compute the MTM P&L & Greeks ---
                last_rec = mtm_records[-1]
                if put_today.empty or call_today.empty:
                    # carry forward prev mtm and greeks
                    pnl_mtm = prev_mtm
                    delta, gamma, vega, theta = (
                        last_rec['delta'], last_rec['gamma'],
                        last_rec['vega'],  last_rec['theta']
                    )
                else:
                    # recompute MTM P&L and Greeks
                    pe_mid = put_side * put_today['mid'].iloc[0]
                    ce_mid = call_side * call_today['mid'].iloc[0]
                    pnl_mtm = (
                        ((pe_mid+ce_mid) * self.lot_size*contracts) -
                          net_entry
                    )
                    delta_pc, gamma_pc, vega_pc, theta_pc = (
                        self._compute_greeks_per_contract(
                            put_today, call_today, put_side, call_side
                        )
                    )
                    delta = delta_pc.iloc[0] * contracts
                    gamma = gamma_pc.iloc[0] * contracts
                    vega  = vega_pc.iloc[0]  * contracts
                    theta = theta_pc.iloc[0] * contracts

                # --- update S and iv ---
                S_curr, iv_curr = last_rec["S"], last_rec["iv"]
                if curr_date in options.index:
                    S_curr  = options.loc[curr_date, 'underlying_last'].iloc[0]
                if curr_date in self.synthetic_skew.index:
                    iv_curr = self.synthetic_skew.loc[curr_date, 'iv_atm_30']

                # --- compute hedge PnL ---
                hedge_price_curr = last_rec['hedge_price_prev']
                if curr_date in self.hedge_series.index:
                    hedge_price_curr = self.hedge_series.loc[curr_date]

                hedge_pnl = last_rec['hedge_qty'] * (
                    hedge_price_curr - last_rec['hedge_price_prev']
                ) * self.hedge_size

                # --- combine net delta and P&L between option and hedge ---
                net_delta = delta + (last_rec['hedge_qty']*self.hedge_size)
                delta_pnl = (pnl_mtm - prev_mtm) + hedge_pnl

                # --- record Market-To-Market ---
                mtm_records.append({
                    'date'             : curr_date,
                    'S'                : S_curr,
                    'iv'               : iv_curr,
                    'delta_pnl'        : delta_pnl,
                    'delta'            : delta,     
                    'net_delta'        : net_delta,
                    'gamma'            : gamma,
                    'vega'             : vega,
                    'theta'            : theta,
                    'hedge_qty'        : last_rec['hedge_qty'], # same
                    'hedge_price_prev' : hedge_price_curr,
                    'hedge_pnl'        : hedge_pnl
                })
                prev_mtm = pnl_mtm

                # --- compute the Realized P&L ---
                if put_today.empty or call_today.empty:
                    continue
                put_exit = (
                    put_today['ask'].iloc[0] + self.slip_ask
                    if sig['short']
                    else put_today['bid'].iloc[0] - self.slip_bid
                )
                call_exit = (
                    call_today['bid'].iloc[0] - self.slip_bid
                    if sig['short']
                    else call_today['ask'].iloc[0] + self.slip_ask
                )
                pnl_per_contract = ((
                    put_side*(put_exit - put_entry) + 
                    call_side*(call_exit - call_entry)
                    ) * self.lot_size
                )
                real_pnl = (pnl_per_contract*contracts) + hedge_pnl

                # --- determine exit type on Realized P&L ---
                exit_type = None
                if real_pnl >= tp_level:
                    exit_type = 'Take Profit'
                elif real_pnl <= sl_level:
                    exit_type = 'Stop Loss'
                elif signals.at[curr_date, 'exit']:
                    exit_type = 'Signal Exit'
                elif curr_date >= hold_date:
                    exit_type = 'Holding Period'

                if not exit_type:
                    continue  # no exit this day, keep looping

                #  --- finalize trade ---
                pnl_net = real_pnl - roundtrip_comm # * contracts
                trades.append({
                    'entry_date'  : entry_date,
                    'exit_date'   : curr_date,
                    'expiry'      : expiry,
                    'contracts'   : contracts,
                    'put_strike'  : put_q['strike'],
                    'call_strike' : call_q['strike'],
                    'put_entry'   : put_entry,
                    'call_entry'  : call_entry,
                    'put_exit'    : put_exit,
                    'call_exit'   : call_exit,
                    'pnl'         : pnl_net,
                    'exit_type'   : exit_type
                })
                # overwrite the last MTM for exit
                mtm_records[-1].update({
                    'delta_pnl' : pnl_net,
                    # for clean daily mtm
                    'delta'     : 0.0,
                    'net_delta' : 0.0,
                    'gamma'     : 0.0,
                    'vega'      : 0.0,
                    'theta'     : 0.0,
                    'hedge_qty' : 0.0
                })
                last_exit = curr_date
                exited = True
                break # go to the next trade

            if not exited:
                del mtm_records[entry_idx:] # skip this trade

        if not mtm_records:
            return pd.DataFrame(trades), pd.DataFrame()
        
        mtm_agg = (
            pd.DataFrame(mtm_records)
            .set_index('date')
            .sort_index()
        )
        mtm = (
            mtm_agg
            .groupby("date")
            .agg({
                'delta_pnl': 'sum',
                'delta'    : 'sum',
                'net_delta': 'sum',
                'gamma'    : 'sum',
                'vega'     : 'sum',
                'theta'    : 'sum',
                'hedge_pnl': 'sum',
                'S'        : 'first',
                'iv'       : 'first',
            })
        )

        # compute equity curve initialized with current capital
        mtm["equity"] = capital + mtm['delta_pnl'].cumsum()

        return pd.DataFrame(trades), mtm
    
    def _rolling_windows(self, dates):
        """Yield contiguous (train_idx, val_idx, test_idx) tuples."""
        i = self.lookback
        while (
            i+self.train_window+self.val_window+self.test_window <=
            len(dates)
        ):
            train_idx = dates[
                i: 
                i+self.train_window
            ]
            val_idx = dates[
                i+self.train_window : 
                i+self.train_window+self.val_window
            ]
            test_idx = dates[
                i+self.train_window+self.val_window :
                i+self.train_window+self.val_window+self.test_window
            ]
            yield train_idx, val_idx, test_idx
            i += self.step


    ##################
    # Public methods #
    ##################

    def walk_forward(self):
        summary, trades_all, mtm_all = [], [], []
        best_params_all = []
        dates = self.options.index.unique()

        for train_idx, val_idx, test_idx in self._rolling_windows(dates):
            # --- pick best params on val window ---
            if self.params_grid is not None:
                best_params = self._score_params(train_idx, val_idx)
                if best_params is not None:
                    self._set_all_params(best_params)
                best_params_all.append({
                    'test_start': test_idx[0],
                    'params':     best_params
                })

            # --- generate signals and apply the filters on test data ---
            signals = self._generate_signals(
                test_idx[0], test_idx[-1]
            )
            signals = self._apply_filters(
                test_idx[0], test_idx[-1], signals
            )

            # --- extract options and signals on test data ---
            sig_test = signals.reindex(test_idx).fillna(False)
            opts_test = self.options.loc[test_idx]

            # --- run the backtest ---
            trades, mtm = self._backtest_risk_reversal(
                opts_test, sig_test, capital=self.current_capital
            )

            # --- store results for walk-forward and backtest ---
            if not trades.empty:
                trades = trades.copy()
                trades['train_start'] = train_idx[0]
                trades['test_start']  = test_idx[0]
                trades_all.append(trades)

            if not mtm.empty:
                # update current capital to last value in eq curve
                self.current_capital = mtm['equity'].iloc[-1]
                mtm = mtm.copy()
                mtm['test_start'] = test_idx[0]
                mtm['test_end'] = test_idx[-1]
                mtm_all.append(mtm)

            summary.append({
                'train_start': train_idx[0],
                'test_start':  test_idx[0],
                'test_pnl':    trades['pnl'].sum() if not trades.empty else 0.0,
                'n_trades':    len(trades),
                'end_equity':  self.current_capital 
            })

        summary_df = pd.DataFrame(summary)
        trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
        mtm_df = pd.concat(mtm_all) if mtm_all else pd.DataFrame()
        params_df = pd.DataFrame(best_params_all)

        return summary_df, trades_df, mtm_df, params_df

    def to_daily_mtm(self, raw_mtm):
        df = raw_mtm.copy()
        df.index = pd.to_datetime(df.index)

        # reindex to calendar days
        full = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df = df.reindex(full)

        # fill P&L and carry-forward Greeks, spot & iv
        df['delta_pnl'] = df['delta_pnl'].fillna(0.0)
        for g in ['net_delta','delta','gamma','vega','theta','S','iv']:
            df[g] = df[g].ffill().fillna(0.0)

        # compute daily moves
        df['dS']     = df['S'].diff().fillna(0.0)
        df['dsigma'] = df['iv'].diff().fillna(0.0)
        df['dt'] = df.index.to_series().diff().dt.days.fillna(1).astype(int)

        # shift prior‐day Greeks
        df['net_delta_prev'] = df['net_delta'].shift(1).fillna(0.0)
        df['delta_prev'] = df['delta'].shift(1).fillna(0.0)
        df['gamma_prev'] = df['gamma'].shift(1).fillna(0.0)
        df['vega_prev']  = df['vega'].shift(1).fillna(0.0)
        df['theta_prev'] = df['theta'].shift(1).fillna(0.0)

        # greek PnL attribution
        df['Delta_PnL'] = df['net_delta_prev'] * df['dS']       # Hedged PnL
        df['Unhedged_Delta_PnL'] = df['delta_prev'] * df['dS'] # Unhedged PnL
        df['Gamma_PnL'] = 0.5 * df['gamma_prev'] * df['dS']**2
        df['Vega_PnL']  = df['vega_prev']  * df['dsigma']
        df['Theta_PnL'] = df['theta_prev'] * df['dt']

        # residual PnL (includes the skew and high order greeks)
        df['Other_PnL'] = df['delta_pnl'] - (
            df['Delta_PnL'] + df['Gamma_PnL']
            + df['Vega_PnL'] + df['Theta_PnL']
        )

        # equity curve as the cumulative PnL
        df['equity'] = self.initial_capital + df['delta_pnl'].cumsum()

        return df