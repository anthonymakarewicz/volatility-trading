"""End-to-end VRP backtest example on processed datasets.

This script demonstrates a minimal pipeline:
1) load processed options + rates datasets,
2) build a VRP strategy spec,
3) configure delta-hedging policy + hedge market feed,
4) run the backtest,
5) print performance metrics.

It assumes the processed datasets already exist under `data/processed/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    BrokerConfig,
    ExecutionConfig,
    HedgeExecutionConfig,
    HedgeMarketData,
    MarginConfig,
    OptionsBacktestDataBundle,
    print_performance_report,
    to_daily_mtm,
)
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    HedgeTriggerPolicy,
)
from volatility_trading.datasets import (
    options_chain_wide_to_long,
    read_fred_rates,
    read_options_chain,
)
from volatility_trading.options import RegTMarginModel
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


@dataclass(frozen=True)
class ExampleConfig:
    """Runtime configuration for the end-to-end VRP example."""

    ticker: str
    start: str
    end: str
    initial_capital: float
    commission_per_leg: float
    rebalance_period: int
    risk_budget_pct: float
    margin_budget_pct: float


def _parse_args() -> ExampleConfig:
    parser = argparse.ArgumentParser(description="Run a minimal VRP E2E backtest.")
    parser.add_argument("--ticker", default="SPY", help="Underlying ticker.")
    parser.add_argument(
        "--start",
        default="2011",
        help="Backtest start date (YYYY or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2017",
        help="Backtest end date (YYYY or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=50_000.0,
        help="Initial capital in USD.",
    )
    parser.add_argument(
        "--commission-per-leg",
        type=float,
        default=0.0,
        help="Commission paid per option leg.",
    )
    parser.add_argument(
        "--rebalance-period",
        type=int,
        default=10,
        help="Rebalance period in calendar days.",
    )
    parser.add_argument(
        "--risk-budget-pct",
        type=float,
        default=1.0,
        help="Risk budget as fraction of equity (0..1+).",
    )
    parser.add_argument(
        "--margin-budget-pct",
        type=float,
        default=0.4,
        help="Initial margin budget as fraction of equity (0..1).",
    )
    args = parser.parse_args()

    return ExampleConfig(
        ticker=str(args.ticker).strip().upper(),
        start=str(args.start),
        end=str(args.end),
        initial_capital=float(args.initial_capital),
        commission_per_leg=float(args.commission_per_leg),
        rebalance_period=int(args.rebalance_period),
        risk_budget_pct=float(args.risk_budget_pct),
        margin_budget_pct=float(args.margin_budget_pct),
    )


def _load_options_long(ticker: str) -> pd.DataFrame:
    """Load processed options chain and convert it to long pandas format."""
    options_wide = read_options_chain(ticker)
    options_long = options_chain_wide_to_long(options_wide).collect().to_pandas()
    options_long["trade_date"] = pd.to_datetime(options_long["trade_date"])
    return options_long.set_index("trade_date").sort_index()


def _load_rf_series(index: pd.DatetimeIndex) -> pd.Series:
    """Load 3M T-bill rate series (DGS3MO) and align it to backtest dates."""
    rf_df = read_fred_rates(columns=["date", "dgs3mo"]).to_pandas()
    rf_df["date"] = pd.to_datetime(rf_df["date"])
    rf = rf_df.set_index("date")["dgs3mo"].astype(float).div(100.0)
    return rf.reindex(index).ffill().fillna(0.0)


def main() -> None:
    cfg = _parse_args()

    options_long = _load_options_long(cfg.ticker)
    options_red = options_long.loc[cfg.start : cfg.end]
    if options_red.empty:
        raise ValueError(
            f"No options rows for ticker={cfg.ticker} in range {cfg.start}:{cfg.end}"
        )

    rf_series = _load_rf_series(pd.DatetimeIndex(options_red.index))
    hedge_mid = options_red.groupby(level=0)["spot_price"].first().astype(float)

    strategy_spec = VRPHarvestingSpec(
        signal=ShortOnlySignal(),
        rebalance_period=cfg.rebalance_period,
        risk_budget_pct=cfg.risk_budget_pct,
        margin_budget_pct=cfg.margin_budget_pct,
        delta_hedge=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                delta_band_abs=25.0,
                rebalance_every_n_days=5,
                combine_mode="or",
            ),
            min_rebalance_qty=1.0,
        ),
    )
    strategy = make_vrp_strategy(strategy_spec)

    backtest_cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=cfg.initial_capital),
        execution=ExecutionConfig(
            commission_per_leg=cfg.commission_per_leg,
            hedge=HedgeExecutionConfig(
                slip_ask=0.0,
                slip_bid=0.0,
                commission_per_unit=0.0,
            ),
        ),
        broker=BrokerConfig(
            margin=MarginConfig(model=RegTMarginModel(broad_index=False))
        ),
    )
    bt = Backtester(
        data=OptionsBacktestDataBundle(
            options=options_red,
            hedge_market=HedgeMarketData(
                mid=hedge_mid,
                symbol=cfg.ticker,
            ),
        ),
        strategy=strategy,
        config=backtest_cfg,
    )
    trades, mtm = bt.run()
    daily_mtm = to_daily_mtm(mtm, backtest_cfg.account.initial_capital)

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


if __name__ == "__main__":
    main()
