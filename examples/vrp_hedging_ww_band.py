"""Focused example: WW-style dynamic no-trade band hedging."""

from __future__ import annotations

from core.cli import parse_common_args
from core.vrp_helpers import build_backtester, load_options_long, run_and_report

from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    HedgeTriggerPolicy,
    WWDeltaBandModel,
)
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


def main() -> None:
    cfg = parse_common_args("Run VRP with WW-style dynamic no-trade band hedging.")

    options_long = load_options_long(cfg.ticker)
    options = options_long.loc[cfg.start : cfg.end]
    if options.empty:
        raise ValueError(
            f"No options rows for {cfg.ticker} in range {cfg.start}:{cfg.end}"
        )

    spec = VRPHarvestingSpec(
        signal=ShortOnlySignal(),
        rebalance_period=10,
        risk_budget_pct=1.0,
        margin_budget_pct=0.4,
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
    )
    strategy = make_vrp_strategy(spec)
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
