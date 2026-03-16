"""Minimal public-API VRP example for running one options backtest.

Run from repository root with:
`python -m examples.backtesting.minimal_vrp_backtest`
"""

from __future__ import annotations

import pandas as pd

from examples.core.backtesting_helpers import load_options_window, load_rf_series
from volatility_trading.backtesting import (
    AccountConfig,
    Backtester,
    BacktestRunConfig,
    BidAskFeeOptionExecutionModel,
    BrokerConfig,
    DeltaHedgePolicy,
    ExecutionConfig,
    FixedBpsHedgeExecutionModel,
    HedgeMarketData,
    MarginConfig,
    MarginPolicy,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    print_performance_report,
    spot_series_from_options_chain,
)
from volatility_trading.options import RegTMarginModel
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy

TICKER = "SPY"
START = "2011-01-01"
END = "2013-12-31"
INITIAL_CAPITAL = 50_000.0


def main() -> None:
    options = load_options_window(ticker=TICKER, start=START, end=END)
    rf_series = load_rf_series(pd.DatetimeIndex(options.index.unique()).sort_values())
    hedge_mid = spot_series_from_options_chain(options)

    data = OptionsBacktestDataBundle(
        options_market=OptionsMarketData(
            chain=options,
            symbol=TICKER,
        ),
        hedge_market=HedgeMarketData(
            mid=hedge_mid,
            symbol=TICKER,
        ),
    )

    strategy = make_vrp_strategy(
        VRPHarvestingSpec(
            signal=ShortOnlySignal(),
            rebalance_period=10,
            risk_budget_pct=1.0,
            margin_budget_pct=0.4,
            delta_hedge=DeltaHedgePolicy(enabled=False),
        )
    )

    config = BacktestRunConfig(
        account=AccountConfig(initial_capital=INITIAL_CAPITAL),
        execution=ExecutionConfig(
            option_execution_model=BidAskFeeOptionExecutionModel(
                commission_per_leg=0.0
            ),
            hedge_execution_model=FixedBpsHedgeExecutionModel(fee_bps=0.0),
        ),
        broker=BrokerConfig(
            margin=MarginConfig(
                model=RegTMarginModel(broad_index=False),
                policy=MarginPolicy(
                    apply_financing=True,
                    cash_rate_annual=rf_series,
                    borrow_rate_annual=rf_series + 0.02,
                ),
            )
        ),
    )

    backtester = Backtester(data=data, strategy=strategy, config=config)
    trades, mtm = backtester.run()

    print(
        f"Minimal example completed: {len(trades)} trades across {len(mtm)} MTM rows."
    )
    print_performance_report(
        trades=trades,
        mtm_daily=mtm,
        risk_free_rate=rf_series,
    )


if __name__ == "__main__":
    main()
