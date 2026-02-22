import pandas as pd

from volatility_trading.backtesting import (
    BacktestConfig,
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
    OpenShortStraddlePosition,
    ShortStraddleEntrySetup,
    ShortStraddleLifecycleEngine,
    SinglePositionRunnerHooks,
    choose_expiry_by_target_dte,
    estimate_short_straddle_margin_per_contract,
    pick_quote_by_delta,
    run_single_position_date_loop,
    size_short_straddle_contracts,
    time_to_expiry_years,
)


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
    ) -> ShortStraddleEntrySetup | None:
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

        return ShortStraddleEntrySetup(
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
        setup: ShortStraddleEntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenShortStraddlePosition, dict]:
        """Create one open-position state and its entry-day MTM record."""
        return self._build_lifecycle_engine().open_position(
            setup=setup,
            cfg=cfg,
            equity_running=equity_running,
        )

    def _mark_open_position(
        self,
        *,
        position: OpenShortStraddlePosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenShortStraddlePosition | None, dict, list[dict]]:
        """Revalue one open position on one date and apply exits/liquidations."""
        return self._build_lifecycle_engine().mark_position(
            position=position,
            curr_date=curr_date,
            options=options,
            cfg=cfg,
            equity_running=equity_running,
        )

    def _build_lifecycle_engine(self) -> ShortStraddleLifecycleEngine:
        """Build lifecycle engine using current strategy parameters."""
        return ShortStraddleLifecycleEngine(
            rebalance_period=self.rebalance_period,
            max_holding_period=self.max_holding_period,
            margin_policy=self.margin_policy,
            margin_model=self.margin_model,
            pricer=self.pricer,
        )

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
        hooks = SinglePositionRunnerHooks[
            OpenShortStraddlePosition, ShortStraddleEntrySetup
        ](
            mark_open_position=lambda position,
            curr_date,
            equity_running: self._mark_open_position(
                position=position,
                curr_date=curr_date,
                options=options,
                cfg=cfg,
                equity_running=equity_running,
            ),
            prepare_entry=lambda entry_date, equity_running: self._prepare_entry_setup(
                entry_date=entry_date,
                options=options,
                features=features,
                equity_running=equity_running,
                cfg=cfg,
            ),
            open_position=lambda setup, equity_running: self._open_position(
                setup=setup,
                cfg=cfg,
                equity_running=equity_running,
            ),
            can_reenter_same_day=self._can_reenter_same_day,
        )
        trades, mtm_records = run_single_position_date_loop(
            trading_dates=trading_dates,
            active_signal_dates=active_signal_dates,
            initial_equity=capital,
            hooks=hooks,
        )

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
