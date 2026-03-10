"""Compare option and hedge execution models under the same strategy.

Run from repository root with:
`python -m examples.backtesting.execution.models_and_costs`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.core.cli import parse_common_args
from examples.core.vrp_helpers import (
    build_backtester,
    build_vrp_strategy,
    load_options_window,
)
from volatility_trading.backtesting import (
    BidAskFeeOptionExecutionModel,
    DeltaHedgePolicy,
    FixedBpsHedgeExecutionModel,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    MidNoCostHedgeExecutionModel,
    MidNoCostOptionExecutionModel,
    compute_performance_metrics,
    to_daily_mtm,
)


@dataclass(frozen=True)
class Scenario:
    """Execution configuration used in one comparison run."""

    name: str
    option_execution_model: object
    hedge_execution_model: object


def main() -> None:
    args = parse_common_args("Compare realistic execution models with no-cost baselines.")
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
            min_rebalance_qty=1.0,
        )
    )
    scenarios = (
        Scenario(
            name="realistic_costs",
            option_execution_model=BidAskFeeOptionExecutionModel(
                commission_per_leg=args.commission_per_leg
            ),
            hedge_execution_model=FixedBpsHedgeExecutionModel(
                fee_bps=args.hedge_fee_bps
            ),
        ),
        Scenario(
            name="no_cost_baseline",
            option_execution_model=MidNoCostOptionExecutionModel(),
            hedge_execution_model=MidNoCostHedgeExecutionModel(),
        ),
    )

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        bt, rf_series, run_cfg = build_backtester(
            options=options,
            ticker=args.ticker,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_per_leg=args.commission_per_leg,
            hedge_fee_bps=args.hedge_fee_bps,
            option_execution_model=scenario.option_execution_model,
            hedge_execution_model=scenario.hedge_execution_model,
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
                "total_pnl": f"{metrics.trades.total_pnl:.2f}",
                "option_trade_cost": f"{mtm['option_trade_cost'].sum():.2f}",
                "hedge_trade_cost": f"{mtm['hedge_trade_cost'].sum():.2f}",
                "sharpe": f"{metrics.returns.sharpe:.4f}",
            }
        )

    print("Execution-model comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"total_pnl={row['total_pnl']}, "
            f"option_trade_cost={row['option_trade_cost']}, "
            f"hedge_trade_cost={row['hedge_trade_cost']}, "
            f"sharpe={row['sharpe']}"
        )


if __name__ == "__main__":
    main()
