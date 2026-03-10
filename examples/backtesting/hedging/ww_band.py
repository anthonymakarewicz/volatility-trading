"""Focused example: WW-style dynamic no-trade band hedging.

Run from repository root with:
`python -m examples.backtesting.hedging.ww_band`
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
    HedgeTriggerPolicy,
    WWDeltaBandModel,
)


def main() -> None:
    cfg = parse_common_args("Run VRP with WW-style dynamic no-trade band hedging.")
    options = load_options_window(ticker=cfg.ticker, start=cfg.start, end=cfg.end)
    strategy = build_vrp_strategy(
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
