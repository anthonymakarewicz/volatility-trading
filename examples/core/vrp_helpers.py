"""Utility helpers shared by backtesting example scripts."""

from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting import (
    AccountConfig,
    Backtester,
    BacktestRunConfig,
    BidAskFeeOptionExecutionModel,
    BrokerConfig,
    DeltaHedgePolicy,
    ExecutionConfig,
    FixedBpsHedgeExecutionModel,
    HedgeExecutionModel,
    HedgeMarketData,
    MarginConfig,
    MarginPolicy,
    OptionExecutionModel,
    OptionsBacktestDataBundle,
    OptionsChainAdapter,
    OptionsMarketData,
    StrategySpec,
    print_performance_report,
    to_daily_mtm,
)
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_fred_rates,
    read_options_chain,
)
from volatility_trading.options import MarginModel, RegTMarginModel
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


def load_options_long(ticker: str) -> pd.DataFrame:
    """Load processed options chain and return a long pandas panel."""
    options_wide = read_options_chain(ticker)
    options_long = options_chain_wide_to_long(options_wide).collect().to_pandas()
    options_long["trade_date"] = pd.to_datetime(options_long["trade_date"])
    return options_long.set_index("trade_date").sort_index()


def load_options_window(*, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load one ticker and return the requested trade-date window."""
    options = load_options_long(ticker).loc[start:end]
    if options.empty:
        raise ValueError(f"No options rows for {ticker} in range {start}:{end}")
    return options


def load_rf_series(index: pd.DatetimeIndex) -> pd.Series:
    """Load 3M T-bill rate series aligned to one backtest date index."""
    rf_df = read_fred_rates(columns=["date", "dgs3mo"]).to_pandas()
    rf_df["date"] = pd.to_datetime(rf_df["date"])
    rf = rf_df.set_index("date")["dgs3mo"].astype(float).div(100.0)
    return rf.reindex(index).ffill().fillna(0.0)


def build_data_bundle(
    *,
    options: pd.DataFrame,
    ticker: str,
    options_adapter: OptionsChainAdapter | None = None,
) -> OptionsBacktestDataBundle:
    """Build one options backtest bundle with spot-based hedge market data."""
    hedge_mid = options.groupby(level=0)["spot_price"].first().astype(float)
    return OptionsBacktestDataBundle(
        options_market=OptionsMarketData(
            chain=options,
            symbol=ticker,
            options_adapter=options_adapter,
        ),
        hedge_market=HedgeMarketData(
            mid=hedge_mid,
            symbol=ticker,
        ),
    )


def build_vrp_strategy(
    *,
    rebalance_period: int = 10,
    risk_budget_pct: float = 1.0,
    margin_budget_pct: float = 0.4,
    delta_hedge: DeltaHedgePolicy | None = None,
) -> StrategySpec:
    """Build one baseline VRP strategy used across examples."""
    return make_vrp_strategy(
        VRPHarvestingSpec(
            signal=ShortOnlySignal(),
            rebalance_period=rebalance_period,
            risk_budget_pct=risk_budget_pct,
            margin_budget_pct=margin_budget_pct,
            delta_hedge=delta_hedge or DeltaHedgePolicy(),
        )
    )


def build_run_config(
    *,
    rf_series: pd.Series,
    initial_capital: float,
    commission_per_leg: float,
    hedge_fee_bps: float,
    hedge_slip_ask: float = 0.0,
    hedge_slip_bid: float = 0.0,
    broad_index: bool = False,
    option_execution_model: OptionExecutionModel | None = None,
    hedge_execution_model: HedgeExecutionModel | None = None,
    margin_model: MarginModel | None = None,
    margin_policy: MarginPolicy | None = None,
) -> BacktestRunConfig:
    """Build one run config with overridable execution and margin components."""
    resolved_margin_policy = margin_policy or MarginPolicy(
        apply_financing=True,
        cash_rate_annual=rf_series,
        borrow_rate_annual=rf_series + 0.02,
    )
    resolved_option_execution_model = option_execution_model or BidAskFeeOptionExecutionModel(
        commission_per_leg=commission_per_leg
    )
    resolved_hedge_execution_model = hedge_execution_model or FixedBpsHedgeExecutionModel(
        slip_ask=hedge_slip_ask,
        slip_bid=hedge_slip_bid,
        fee_bps=hedge_fee_bps,
    )
    resolved_margin_model = margin_model or RegTMarginModel(broad_index=broad_index)

    return BacktestRunConfig(
        account=AccountConfig(initial_capital=initial_capital),
        execution=ExecutionConfig(
            option_execution_model=resolved_option_execution_model,
            hedge_execution_model=resolved_hedge_execution_model,
        ),
        broker=BrokerConfig(
            margin=MarginConfig(
                model=resolved_margin_model,
                policy=resolved_margin_policy,
            )
        ),
    )


def build_backtester(
    *,
    options: pd.DataFrame,
    ticker: str,
    strategy: StrategySpec,
    initial_capital: float,
    commission_per_leg: float,
    hedge_fee_bps: float,
    hedge_slip_ask: float = 0.0,
    hedge_slip_bid: float = 0.0,
    options_adapter: OptionsChainAdapter | None = None,
    option_execution_model: OptionExecutionModel | None = None,
    hedge_execution_model: HedgeExecutionModel | None = None,
    margin_model: MarginModel | None = None,
    margin_policy: MarginPolicy | None = None,
    broad_index: bool = False,
) -> tuple[Backtester, pd.Series, BacktestRunConfig]:
    """Build a configured backtester plus one aligned risk-free series."""
    rf_series = load_rf_series(pd.DatetimeIndex(options.index.unique()).sort_values())
    cfg = build_run_config(
        rf_series=rf_series,
        initial_capital=initial_capital,
        commission_per_leg=commission_per_leg,
        hedge_fee_bps=hedge_fee_bps,
        hedge_slip_ask=hedge_slip_ask,
        hedge_slip_bid=hedge_slip_bid,
        option_execution_model=option_execution_model,
        hedge_execution_model=hedge_execution_model,
        margin_model=margin_model,
        margin_policy=margin_policy,
        broad_index=broad_index,
    )
    bt = Backtester(
        data=build_data_bundle(
            options=options,
            ticker=ticker,
            options_adapter=options_adapter,
        ),
        strategy=strategy,
        config=cfg,
    )
    return bt, rf_series, cfg


def run_and_report(
    *,
    backtester: Backtester,
    initial_capital: float,
    rf_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one backtest and print standard performance output."""
    trades, mtm = backtester.run()
    daily_mtm = to_daily_mtm(mtm, initial_capital)
    print(
        "Backtest completed: "
        f"{len(trades)} trades, "
        f"{len(daily_mtm)} MTM rows, "
        f"period={daily_mtm.index.min().date()} to {daily_mtm.index.max().date()}"
    )
    print_performance_report(
        trades=trades,
        mtm_daily=daily_mtm,
        risk_free_rate=rf_series,
    )
    return trades, daily_mtm
