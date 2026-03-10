"""Focused example: compare realistic bps costs vs zero-cost baseline."""

from __future__ import annotations

from dataclasses import dataclass

from core.cli import parse_common_args
from core.vrp_helpers import build_backtester, load_options_long

from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    compute_performance_metrics,
    to_daily_mtm,
)
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


@dataclass(frozen=True)
class Scenario:
    """Hedge execution scenario used in comparison output."""

    name: str
    hedge_fee_bps: float
    hedge_slip_ask: float
    hedge_slip_bid: float


def _build_strategy():
    spec = VRPHarvestingSpec(
        signal=ShortOnlySignal(),
        rebalance_period=10,
        risk_budget_pct=1.0,
        margin_budget_pct=0.4,
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
    )
    return make_vrp_strategy(spec)


def main() -> None:
    args = parse_common_args(
        "Compare VRP hedge execution cost scenarios (fixed bps vs zero-cost baseline)."
    )

    options_long = load_options_long(args.ticker)
    options = options_long.loc[args.start : args.end]
    if options.empty:
        raise ValueError(
            f"No options rows for {args.ticker} in range {args.start}:{args.end}"
        )
    strategy = _build_strategy()
    scenarios = (
        Scenario(
            name="fixed_bps_costs",
            hedge_fee_bps=args.hedge_fee_bps,
            hedge_slip_ask=0.0,
            hedge_slip_bid=0.0,
        ),
        # High-level backtester baseline approximation of no-cost hedge execution.
        Scenario(
            name="zero_cost_baseline",
            hedge_fee_bps=0.0,
            hedge_slip_ask=0.0,
            hedge_slip_bid=0.0,
        ),
    )

    rows: list[dict[str, float | str | None]] = []
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
        daily_mtm = to_daily_mtm(mtm, run_cfg.account.initial_capital)
        metrics = compute_performance_metrics(
            trades=trades,
            mtm_daily=daily_mtm,
            risk_free_rate=rf_series,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "total_return": metrics.returns.total_return,
                "sharpe": metrics.returns.sharpe,
                "max_drawdown": metrics.drawdown.max_drawdown,
                "total_pnl": metrics.trades.total_pnl,
                "trade_count": float(metrics.trades.total_trades),
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
