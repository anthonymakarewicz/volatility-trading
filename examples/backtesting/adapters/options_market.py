"""Show how `OptionsMarketData` scopes adapter configuration per dataset.

Run from repository root with:
`python -m examples.backtesting.adapters.options_market`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.core.cli import parse_common_args
from examples.core.vrp_helpers import (
    build_run_config,
    build_vrp_strategy,
    load_options_window,
    load_rf_series,
)
from volatility_trading.backtesting import (
    Backtester,
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    HedgeMarketData,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    compute_performance_metrics,
    to_daily_mtm,
)


@dataclass(frozen=True)
class Scenario:
    """Adapter scenario used in comparison output."""

    name: str
    options_market: OptionsMarketData


def main() -> None:
    args = parse_common_args("Compare canonical and mapped options-chain adapters.")
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    rf_series = load_rf_series(options.index.unique())
    hedge_market = HedgeMarketData(
        mid=options.groupby(level=0)["spot_price"].first().astype(float),
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
                chain=options,
                symbol=args.ticker,
                options_adapter=CanonicalOptionsChainAdapter(),
            ),
        ),
        Scenario(
            name="column_map_adapter",
            options_market=OptionsMarketData(
                chain=aliased_options,
                symbol=args.ticker,
                options_adapter=ColumnMapOptionsChainAdapter(
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

    print("OptionsMarketData adapter comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"trade_count={row['trade_count']}, "
            f"total_pnl={row['total_pnl']}, "
            f"total_return={row['total_return']}"
        )


if __name__ == "__main__":
    main()
