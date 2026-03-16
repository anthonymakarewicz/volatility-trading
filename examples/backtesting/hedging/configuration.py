"""Compare common delta-hedging configurations under the same strategy.

Run from repository root with:
`python -m examples.backtesting.hedging.configuration`
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
    WWDeltaBandModel,
    compute_performance_metrics,
)


@dataclass(frozen=True)
class Scenario:
    """Delta-hedging policy scenario used in comparison output."""

    name: str
    delta_hedge: DeltaHedgePolicy


def main() -> None:
    args = parse_common_args("Compare disabled, fixed-band, and WW hedging policies.")
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    scenarios = (
        Scenario(name="hedging_disabled", delta_hedge=DeltaHedgePolicy(enabled=False)),
        Scenario(
            name="fixed_band",
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
            ),
        ),
        Scenario(
            name="ww_band",
            delta_hedge=DeltaHedgePolicy(
                enabled=True,
                target_net_delta=0.0,
                trigger=HedgeTriggerPolicy(
                    band_model=WWDeltaBandModel(
                        calibration_c=1.0,
                        min_band_abs=5.0,
                        max_band_abs=40.0,
                    ),
                    rebalance_every_n_days=None,
                    combine_mode="or",
                ),
                rebalance_to="nearest_boundary",
                min_rebalance_qty=1.0,
            ),
        ),
    )

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        strategy = build_vrp_strategy(delta_hedge=scenario.delta_hedge)
        bt, rf_series, run_cfg = build_backtester(
            options=options,
            ticker=args.ticker,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_per_leg=args.commission_per_leg,
            hedge_fee_bps=args.hedge_fee_bps,
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
                "total_pnl": f"{metrics.trades.total_pnl:.2f}",
                "hedge_trade_cost": f"{mtm['hedge_trade_cost'].sum():.2f}",
                "hedge_trade_count": str(int(mtm["hedge_trade_count"].sum())),
                "sharpe": f"{metrics.returns.sharpe:.4f}",
            }
        )

    print("Delta-hedging configuration comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"total_pnl={row['total_pnl']}, "
            f"hedge_trade_cost={row['hedge_trade_cost']}, "
            f"hedge_trade_count={row['hedge_trade_count']}, "
            f"sharpe={row['sharpe']}"
        )


if __name__ == "__main__":
    main()
