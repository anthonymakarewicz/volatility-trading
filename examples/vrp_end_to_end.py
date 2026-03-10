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

from core.cli import parse_vrp_args
from core.vrp_helpers import build_backtester, load_options_long, run_and_report

from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
)
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


def main() -> None:
    cfg = parse_vrp_args("Run a minimal VRP E2E backtest.")

    options_long = load_options_long(cfg.ticker)
    options_red = options_long.loc[cfg.start : cfg.end]
    if options_red.empty:
        raise ValueError(
            f"No options rows for ticker={cfg.ticker} in range {cfg.start}:{cfg.end}"
        )

    strategy_spec = VRPHarvestingSpec(
        signal=ShortOnlySignal(),
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
    strategy = make_vrp_strategy(strategy_spec)
    bt, rf_series, _ = build_backtester(
        options=options_red,
        ticker=cfg.ticker,
        strategy=strategy,
        initial_capital=cfg.initial_capital,
        commission_per_leg=cfg.commission_per_leg,
        hedge_fee_bps=cfg.hedge_fee_bps,
        hedge_slip_ask=0.0,
        hedge_slip_bid=0.0,
    )
    run_and_report(
        backtester=bt,
        initial_capital=cfg.initial_capital,
        rf_series=rf_series,
    )


if __name__ == "__main__":
    main()
