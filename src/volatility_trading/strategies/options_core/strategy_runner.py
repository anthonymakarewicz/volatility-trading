"""Generic options strategy runner for single-position backtests."""

from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting import BacktestConfig, SliceContext

from .entry import build_entry_intent_from_structure, normalize_signals_to_on
from .lifecycle import (
    OpenPosition,
    PositionEntrySetup,
    PositionLifecycleEngine,
)
from .runner import SinglePositionRunnerHooks, run_single_position_date_loop
from .sizing import size_entry_intent_contracts
from .specs import StrategySpec


class OptionsStrategyRunner:
    """Top-level orchestrator that executes one `StrategySpec`."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec
        self.signal = spec.signal
        self.filters = list(spec.filters)

        # Keep direct attributes for observability/reporting.
        self.structure_spec = spec.structure_spec
        self.rebalance_period = spec.rebalance_period
        self.max_holding_period = spec.max_holding_period
        self.exit_rule_set = spec.exit_rule_set
        self.reentry_policy = spec.reentry_policy
        self.pricer = spec.pricer
        self.scenario_generator = spec.scenario_generator
        self.risk_estimator = spec.risk_estimator
        self.risk_sizer = spec.risk_sizer
        self.margin_model = spec.margin_model
        self.margin_budget_pct = spec.margin_budget_pct
        self.margin_policy = spec.margin_policy
        self.min_contracts = spec.min_contracts
        self.max_contracts = spec.max_contracts

    def run(self, ctx: SliceContext):
        data = ctx.data
        capital = ctx.capital
        cfg = ctx.config

        options = data[self.spec.options_data_key]
        features = data.get(self.spec.features_data_key)
        hedge = data.get(self.spec.hedge_data_key)

        signal_input = self.spec.signal_input_builder(options, features, hedge)
        signals = self.signal.generate_signals(signal_input)

        if self.filters:
            filter_ctx = self.spec.filter_context_builder(options, features, hedge)
            for filt in self.filters:
                signals = filt.apply(signals, filter_ctx)

        trades, mtm = self._simulate_structure(
            options=options,
            signals=signals,
            features=features,
            hedge=hedge,
            capital=capital,
            cfg=cfg,
        )
        return trades, mtm

    def _signals_to_on_frame(self, signals: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Normalize generated signals into boolean `on` events."""
        return normalize_signals_to_on(signals, strategy_name=self.spec.name)

    def _prepare_entry_setup(
        self,
        *,
        entry_date: pd.Timestamp,
        entry_direction: int,
        options: pd.DataFrame,
        features: pd.DataFrame | None,
        equity_running: float,
        cfg: BacktestConfig,
    ) -> PositionEntrySetup | None:
        """Build one structure entry setup for a specific date."""
        if entry_direction == 0:
            return None

        intent = build_entry_intent_from_structure(
            entry_date=entry_date,
            options=options,
            structure_spec=self.structure_spec,
            cfg=cfg,
            side_resolver=lambda leg_spec: self.spec.side_resolver(
                leg_spec,
                entry_direction,
            ),
            features=features,
            fallback_iv_feature_col=self.spec.fallback_iv_feature_col,
        )
        if intent is None:
            return None

        spot_entry = float(intent.spot) if intent.spot is not None else float("nan")
        iv_entry = (
            float(intent.volatility) if intent.volatility is not None else float("nan")
        )
        contracts, risk_pc, risk_scenario, margin_pc = size_entry_intent_contracts(
            intent=intent,
            lot_size=cfg.lot_size,
            spot=spot_entry,
            volatility=iv_entry,
            equity=float(equity_running),
            pricer=self.pricer,
            scenario_generator=self.scenario_generator,
            risk_estimator=self.risk_estimator,
            risk_sizer=self.risk_sizer,
            margin_model=self.margin_model,
            margin_budget_pct=self.margin_budget_pct,
            min_contracts=self.min_contracts,
            max_contracts=self.max_contracts,
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

    def _build_lifecycle_engine(self) -> PositionLifecycleEngine:
        return PositionLifecycleEngine(
            rebalance_period=self.rebalance_period,
            max_holding_period=self.max_holding_period,
            exit_rule_set=self.exit_rule_set,
            margin_policy=self.margin_policy,
            margin_model=self.margin_model,
            pricer=self.pricer,
        )

    def _can_reenter_same_day(self, trade_rows: list[dict]) -> bool:
        return self.reentry_policy.allow_from_trade_rows(trade_rows)

    def _simulate_structure(
        self,
        options: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None,
        hedge: pd.Series | pd.DataFrame | None,
        capital: float,
        cfg: BacktestConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _ = hedge

        sig_df = self._signals_to_on_frame(signals)
        options = options.sort_index()
        trading_dates = sorted(options.index.unique())
        direction_by_date = (
            sig_df["entry_direction"].groupby(level=0).last().astype(int)
        )
        active_signal_dates = set(direction_by_date[direction_by_date != 0].index)
        lifecycle_engine = self._build_lifecycle_engine()
        hooks = SinglePositionRunnerHooks[OpenPosition, PositionEntrySetup](
            mark_open_position=lambda position, curr_date, equity_running: (
                lifecycle_engine.mark_position(
                    position=position,
                    curr_date=curr_date,
                    options=options,
                    cfg=cfg,
                    equity_running=equity_running,
                )
            ),
            prepare_entry=lambda entry_date, equity_running: self._prepare_entry_setup(
                entry_date=entry_date,
                entry_direction=int(direction_by_date.get(entry_date, 0)),
                options=options,
                features=features,
                equity_running=equity_running,
                cfg=cfg,
            ),
            open_position=lambda setup, equity_running: lifecycle_engine.open_position(
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
