"""Focused example: compare realistic hedge costs vs a zero-cost baseline.

Run from repository root with:
`python -m examples.backtesting.hedging.cost_baselines`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.backtesting.strategies import build_vrp_strategy
from examples.core.backtesting_helpers import build_backtester, load_options_window
from examples.core.cli import parse_common_args
from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    compute_performance_metrics,
)


@dataclass(frozen=True)
class Scenario:
    """Hedge execution scenario used in comparison output."""

    name: str
    hedge_fee_bps: float
    hedge_slip_ask: float
    hedge_slip_bid: float


def main() -> None:
    args = parse_common_args(
        "Compare VRP hedge execution cost scenarios (fixed bps vs zero-cost baseline)."
    )
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    strategy = build_vrp_strategy(
        delta_hedge=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=25.0),
                rebalance_every_n_days=5,
                combine_mode="or",
            ),
            rebalance_to="center",
            min_rebalance_qty=1.0,
        )
    )
    scenarios = (
        Scenario(
            name="fixed_bps_costs",
            hedge_fee_bps=args.hedge_fee_bps,
            hedge_slip_ask=0.0,
            hedge_slip_bid=0.0,
        ),
        Scenario(
            name="zero_cost_baseline",
            hedge_fee_bps=0.0,
            hedge_slip_ask=0.0,
            hedge_slip_bid=0.0,
        ),
    )

    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        bt, rf_series, run_cfg = build_backtester(
            options=options,
            ticker=args.ticker,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_per_leg=args.commission_per_leg,
            hedge_fee_bps=scenario.hedge_fee_bps,
            hedge_slip_ask=scenario.hedge_slip_ask,
            hedge_slip_bid=scenario.hedge_slip_bid,
        )
        trades, mtm = bt.run()
        metrics = compute_performance_metrics(
            trades=trades,
            mtm_daily=mtm,
            risk_free_rate=rf_series,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "total_return": f"{metrics.returns.total_return:.4f}",
                "sharpe": f"{metrics.returns.sharpe:.4f}",
                "max_drawdown": f"{metrics.drawdown.max_drawdown:.4f}",
                "total_pnl": f"{metrics.trades.total_pnl:.2f}",
                "trade_count": str(metrics.trades.total_trades),
            }
        )

    print("Hedge cost scenario comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"total_return={row['total_return']}, "
            f"sharpe={row['sharpe']}, "
            f"max_drawdown={row['max_drawdown']}, "
            f"total_pnl={row['total_pnl']}, "
            f"trade_count={row['trade_count']}"
        )


if __name__ == "__main__":
    main()
