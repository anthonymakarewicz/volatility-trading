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
    OptionType,
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
    EntryIntent,
    ExitRuleSet,
    LegSpec,
    OpenPosition,
    PositionEntrySetup,
    PositionLifecycleEngine,
    SameDayReentryPolicy,
    SinglePositionRunnerHooks,
    StructureSpec,
    build_entry_intent_from_structure,
    choose_expiry_by_target_dte,
    normalize_signals_to_on,
    pick_quote_by_delta,
    run_single_position_date_loop,
    size_entry_intent_contracts,
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
        exit_rule_set: ExitRuleSet | None = None,
        reentry_policy: SameDayReentryPolicy | None = None,
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
        self.reentry_policy = reentry_policy or SameDayReentryPolicy(
            allow_on_rebalance=allow_same_day_reentry_on_rebalance,
            allow_on_max_holding=allow_same_day_reentry_on_max_holding,
        )
        self.exit_rule_set = exit_rule_set or ExitRuleSet.period_rules()
        self.target_dte = target_dte
        self.max_dte_diff = max_dte_diff
        self.structure_spec = StructureSpec(
            name="short_atm_straddle",
            dte_target=target_dte,
            dte_tolerance=max_dte_diff,
            legs=(
                LegSpec(option_type=OptionType.PUT, delta_target=-0.5),
                LegSpec(option_type=OptionType.CALL, delta_target=0.5),
            ),
        )
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

    def _size_entry_intent(
        self,
        *,
        intent: EntryIntent,
        lot_size: int,
        spot: float,
        volatility: float,
        equity: float,
    ) -> tuple[int, float | None, str | None, float | None]:
        """Compute contracts from risk budget and optional margin-capacity budget."""
        return size_entry_intent_contracts(
            intent=intent,
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

        trades, mtm = self._simulate_structure(
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
        """Normalize signals using the shared options-core signal adapter."""
        return normalize_signals_to_on(signals, strategy_name="VRPStrategy")

    @staticmethod
    def _entry_side_for_leg(_leg_spec: LegSpec) -> int:
        """VRP baseline enters as net short-vol: short every selected leg."""
        return -1

    def _prepare_entry_setup(
        self,
        *,
        entry_date: pd.Timestamp,
        options: pd.DataFrame,
        features: pd.DataFrame | None,
        equity_running: float,
        cfg: BacktestConfig,
    ) -> PositionEntrySetup | None:
        """Build an entry setup from `self.structure_spec` plus sizing constraints."""
        intent = build_entry_intent_from_structure(
            entry_date=entry_date,
            options=options,
            structure_spec=self.structure_spec,
            cfg=cfg,
            side_resolver=self._entry_side_for_leg,
            features=features,
        )
        if intent is None:
            return None

        spot_entry = float(intent.spot) if intent.spot is not None else float("nan")
        iv_entry = (
            float(intent.volatility) if intent.volatility is not None else float("nan")
        )
        contracts, risk_pc, risk_scenario, margin_pc = self._size_entry_intent(
            intent=intent,
            lot_size=cfg.lot_size,
            spot=spot_entry,
            volatility=iv_entry,
            equity=float(equity_running),
        )
        if contracts <= 0:
            return None

        return PositionEntrySetup(
            intent=intent,
            contracts=int(contracts),
            risk_per_contract=risk_pc,
            risk_worst_scenario=risk_scenario,
            margin_per_contract=margin_pc,
        )

    def _open_position(
        self,
        *,
        setup: PositionEntrySetup,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition, dict]:
        """Create one open-position state and its entry-day MTM record."""
        return self._build_lifecycle_engine().open_position(
            setup=setup,
            cfg=cfg,
            equity_running=equity_running,
        )

    def _mark_open_position(
        self,
        *,
        position: OpenPosition,
        curr_date: pd.Timestamp,
        options: pd.DataFrame,
        cfg: BacktestConfig,
        equity_running: float,
    ) -> tuple[OpenPosition | None, dict, list[dict]]:
        """Revalue one open position on one date and apply exits/liquidations."""
        return self._build_lifecycle_engine().mark_position(
            position=position,
            curr_date=curr_date,
            options=options,
            cfg=cfg,
            equity_running=equity_running,
        )

    def _build_lifecycle_engine(self) -> PositionLifecycleEngine:
        """Build lifecycle engine using current strategy parameters."""
        return PositionLifecycleEngine(
            rebalance_period=self.rebalance_period,
            max_holding_period=self.max_holding_period,
            exit_rule_set=self.exit_rule_set,
            margin_policy=self.margin_policy,
            margin_model=self.margin_model,
            pricer=self.pricer,
        )

    def _can_reenter_same_day(self, trade_rows: list[dict]) -> bool:
        """Allow same-day reentry according to exit-specific policy."""
        return self.reentry_policy.allow_from_trade_rows(trade_rows)

    def _simulate_structure(
        self,
        options: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None,
        hedge: pd.Series | None,
        capital: float,
        cfg: BacktestConfig,
    ):
        """Run a single-position-at-a-time, date-driven structure simulation."""
        _ = hedge

        sig_df = self._signals_to_on_frame(signals)
        options = options.sort_index()
        trading_dates = sorted(options.index.unique())
        active_signal_dates = set(sig_df.index[sig_df["on"]])
        hooks = SinglePositionRunnerHooks[OpenPosition, PositionEntrySetup](
            mark_open_position=lambda position, curr_date, equity_running: (
                self._mark_open_position(
                    position=position,
                    curr_date=curr_date,
                    options=options,
                    cfg=cfg,
                    equity_running=equity_running,
                )
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
