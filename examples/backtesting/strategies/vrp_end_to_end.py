"""End-to-end VRP backtest example on processed datasets.

This script demonstrates a minimal pipeline:
1) load processed options + rates datasets,
2) build a VRP strategy spec,
3) configure delta hedging,
4) run the backtest,
5) print performance metrics.

Run from repository root with:
`python -m examples.backtesting.strategies.vrp_end_to_end`
"""

from __future__ import annotations

from examples.backtesting.strategies import build_vrp_strategy
from examples.core.backtesting_helpers import (
    build_backtester,
    load_options_window,
    run_and_report,
)
from examples.core.cli import parse_vrp_args
from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
)


def main() -> None:
    cfg = parse_vrp_args("Run a minimal VRP E2E backtest.")
    options = load_options_window(ticker=cfg.ticker, start=cfg.start, end=cfg.end)

    strategy = build_vrp_strategy(
        rebalance_period=cfg.rebalance_period,
        risk_budget_pct=cfg.risk_budget_pct,
        margin_budget_pct=cfg.margin_budget_pct,
        delta_hedge=DeltaHedgePolicy(
            enabled=True,
            target_net_delta=0.0,
            trigger=HedgeTriggerPolicy(
                band_model=FixedDeltaBandModel(half_width_abs=25.0),
                rebalance_every_n_days=5,
                combine_mode="or",
            ),
            min_rebalance_qty=1.0,
        ),
    )
    bt, rf_series, _ = build_backtester(
        options=options,
        ticker=cfg.ticker,
        strategy=strategy,
        initial_capital=cfg.initial_capital,
        commission_per_leg=cfg.commission_per_leg,
        hedge_fee_bps=cfg.hedge_fee_bps,
    )
    run_and_report(
        backtester=bt,
        initial_capital=cfg.initial_capital,
        rf_series=rf_series,
    )


if __name__ == "__main__":
    main()
