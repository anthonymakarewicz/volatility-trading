"""VRP example showing structure-level stop-loss / take-profit exits.

This script demonstrates the current richer-exits v1 surface:
1) load processed options + rates datasets,
2) build a VRP preset with `pnl_per_contract` stop-loss / take-profit exits,
3) run the backtest,
4) print performance metrics.

Run from repository root with:
`python -m examples.backtesting.exits.pnl_per_contract`
"""

from __future__ import annotations

from examples.core.backtesting_helpers import (
    build_backtester,
    load_options_window,
    run_and_report,
)
from examples.core.cli import parse_vrp_args
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy


def main() -> None:
    cfg = parse_vrp_args("Run a VRP backtest with richer exit rules.")
    options = load_options_window(ticker=cfg.ticker, start=cfg.start, end=cfg.end)

    strategy = make_vrp_strategy(
        VRPHarvestingSpec(
            signal=ShortOnlySignal(),
            rebalance_period=cfg.rebalance_period,
            risk_budget_pct=cfg.risk_budget_pct,
            margin_budget_pct=cfg.margin_budget_pct,
            stop_loss_pnl_per_contract=1.0,
            take_profit_pnl_per_contract=1.5,
            allow_same_day_reentry_on_rebalance=False,
            allow_same_day_reentry_on_max_holding=False,
            allow_same_day_reentry_on_stop_loss=False,
            allow_same_day_reentry_on_take_profit=False,
        )
    )
    print(
        "Configured richer exits: "
        "stop_loss_pnl_per_contract=1.0, "
        "take_profit_pnl_per_contract=1.5"
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
