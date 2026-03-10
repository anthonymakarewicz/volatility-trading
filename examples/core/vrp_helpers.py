"""Utility helpers shared by VRP example scripts."""

from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting import (
    AccountConfig,
    Backtester,
    BacktestRunConfig,
    BidAskFeeOptionExecutionModel,
    BrokerConfig,
    ExecutionConfig,
    FixedBpsHedgeExecutionModel,
    HedgeMarketData,
    MarginConfig,
    MarginPolicy,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    print_performance_report,
    to_daily_mtm,
)
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_fred_rates,
    read_options_chain,
)
from volatility_trading.options import RegTMarginModel


def load_options_long(ticker: str) -> pd.DataFrame:
    """Load processed options chain and return a long pandas panel."""
    options_wide = read_options_chain(ticker)
    options_long = options_chain_wide_to_long(options_wide).collect().to_pandas()
    options_long["trade_date"] = pd.to_datetime(options_long["trade_date"])
    return options_long.set_index("trade_date").sort_index()


def load_rf_series(index: pd.DatetimeIndex) -> pd.Series:
    """Load 3M T-bill rate series aligned to one backtest date index."""
    rf_df = read_fred_rates(columns=["date", "dgs3mo"]).to_pandas()
    rf_df["date"] = pd.to_datetime(rf_df["date"])
    rf = rf_df.set_index("date")["dgs3mo"].astype(float).div(100.0)
    return rf.reindex(index).ffill().fillna(0.0)


def build_data_bundle(*, options: pd.DataFrame, ticker: str) -> OptionsBacktestDataBundle:
    """Build one options backtest bundle with a spot-based hedge mid series."""
    hedge_mid = options.groupby(level=0)["spot_price"].first().astype(float)
    return OptionsBacktestDataBundle(
        options_market=OptionsMarketData(
            chain=options,
            symbol=ticker,
        ),
        hedge_market=HedgeMarketData(
            mid=hedge_mid,
            symbol=ticker,
        ),
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
) -> BacktestRunConfig:
    """Build one run config for examples with Reg-T margin and financing."""
    margin_policy = MarginPolicy(
        apply_financing=True,
        cash_rate_annual=rf_series,
        borrow_rate_annual=rf_series + 0.02,
    )
    return BacktestRunConfig(
        account=AccountConfig(initial_capital=initial_capital),
        execution=ExecutionConfig(
            option_execution_model=BidAskFeeOptionExecutionModel(
                commission_per_leg=commission_per_leg
            ),
            hedge_execution_model=FixedBpsHedgeExecutionModel(
                slip_ask=hedge_slip_ask,
                slip_bid=hedge_slip_bid,
                fee_bps=hedge_fee_bps,
            ),
        ),
        broker=BrokerConfig(
            margin=MarginConfig(model=RegTMarginModel(broad_index=broad_index), policy=margin_policy)
        ),
    )


def build_backtester(
    *,
    options: pd.DataFrame,
    ticker: str,
    strategy,
    initial_capital: float,
    commission_per_leg: float,
    hedge_fee_bps: float,
    hedge_slip_ask: float = 0.0,
    hedge_slip_bid: float = 0.0,
) -> tuple[Backtester, pd.Series, BacktestRunConfig]:
    """Build a configured backtester plus aligned risk-free series."""
    rf_series = load_rf_series(pd.DatetimeIndex(options.index))
    cfg = build_run_config(
        rf_series=rf_series,
        initial_capital=initial_capital,
        commission_per_leg=commission_per_leg,
        hedge_fee_bps=hedge_fee_bps,
        hedge_slip_ask=hedge_slip_ask,
        hedge_slip_bid=hedge_slip_bid,
    )
    bt = Backtester(
        data=build_data_bundle(options=options, ticker=ticker),
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
