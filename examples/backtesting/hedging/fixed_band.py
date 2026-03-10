"""Focused example: fixed-band delta hedging with periodic trigger.

Run from repository root with:
`python -m examples.backtesting.hedging.fixed_band`
"""

from __future__ import annotations

from examples.backtesting.strategies import build_vrp_strategy
from examples.core.backtesting_helpers import (
    build_backtester,
    load_options_window,
    run_and_report,
)
from examples.core.cli import parse_common_args
from volatility_trading.backtesting import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
)


def main() -> None:
    cfg = parse_common_args("Run VRP with fixed-band delta hedging.")
    options = load_options_window(ticker=cfg.ticker, start=cfg.start, end=cfg.end)
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
