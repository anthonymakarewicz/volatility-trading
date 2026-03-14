"""Show explicit options-chain canonicalization before `OptionsMarketData`.

Run from repository root with:
`python -m examples.backtesting.adapters.options_market`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.backtesting.strategies import build_vrp_strategy
from examples.core.backtesting_helpers import (
    build_run_config,
    load_options_window,
    load_rf_series,
)
from examples.core.cli import parse_common_args
from volatility_trading.backtesting import (
    Backtester,
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    HedgeMarketData,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    canonicalize_options_chain_for_backtest,
    compute_performance_metrics,
    spot_series_from_options_chain,
    to_daily_mtm,
)


@dataclass(frozen=True)
class Scenario:
    """Adapter scenario used in comparison output."""

    name: str
    options_market: OptionsMarketData


def main() -> None:
    args = parse_common_args("Compare explicit canonicalization paths for options data.")
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    rf_series = load_rf_series(options.index.unique())
    hedge_market = HedgeMarketData(
        mid=spot_series_from_options_chain(options),
        symbol=args.ticker,
    )
    aliased_options = options.reset_index().rename(
        columns={
            "trade_date": "qdt",
            "expiry_date": "exp",
            "dte": "days",
            "option_type": "cp",
            "strike": "k",
            "delta": "d",
            "bid_price": "b",
            "ask_price": "a",
        }
    )
    scenarios = (
        Scenario(
            name="canonical_adapter",
            options_market=OptionsMarketData(
                chain=canonicalize_options_chain_for_backtest(
                    options,
                    adapter=CanonicalOptionsChainAdapter(),
                ),
                symbol=args.ticker,
            ),
        ),
        Scenario(
            name="column_map_adapter",
            options_market=OptionsMarketData(
                chain=canonicalize_options_chain_for_backtest(
                    aliased_options,
                    adapter=ColumnMapOptionsChainAdapter(
                        source_to_canonical={
                            "qdt": "trade_date",
                            "exp": "expiry_date",
                            "days": "dte",
                            "cp": "option_type",
                            "k": "strike",
                            "d": "delta",
                            "b": "bid_price",
                            "a": "ask_price",
                        }
                    ),
                ),
                symbol=args.ticker,
            ),
        ),
    )

    strategy = build_vrp_strategy()
    run_cfg = build_run_config(
        rf_series=rf_series,
        initial_capital=args.initial_capital,
        commission_per_leg=args.commission_per_leg,
        hedge_fee_bps=args.hedge_fee_bps,
    )

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        bt = Backtester(
            data=OptionsBacktestDataBundle(
                options_market=scenario.options_market,
                hedge_market=hedge_market,
            ),
            strategy=strategy,
            config=run_cfg,
        )
        trades, mtm = bt.run()
        daily_mtm = to_daily_mtm(mtm, run_cfg.account.initial_capital)
        metrics = compute_performance_metrics(
            trades=trades,
            mtm_daily=daily_mtm,
            risk_free_rate=rf_series,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "trade_count": str(metrics.trades.total_trades),
                "total_pnl": f"{metrics.trades.total_pnl:.2f}",
                "total_return": f"{metrics.returns.total_return:.4f}",
            }
        )

    print("Explicit canonicalization comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"trade_count={row['trade_count']}, "
            f"total_pnl={row['total_pnl']}, "
            f"total_return={row['total_return']}"
        )


if __name__ == "__main__":
    main()
