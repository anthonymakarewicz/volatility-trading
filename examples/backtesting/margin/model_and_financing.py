"""Compare margin-model and financing configurations.

Run from repository root with:
`python -m examples.backtesting.margin.model_and_financing`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.backtesting.strategies import build_vrp_strategy
from examples.core.backtesting_helpers import (
    build_backtester,
    load_options_window,
    load_rf_series,
)
from examples.core.cli import parse_common_args
from volatility_trading.backtesting import (
    MarginPolicy,
    compute_performance_metrics,
)
from volatility_trading.options import RegTMarginModel


@dataclass(frozen=True)
class Scenario:
    """Margin-model and financing setup used in one run."""

    name: str
    margin_model: RegTMarginModel
    margin_policy: MarginPolicy


def main() -> None:
    args = parse_common_args("Compare margin-model and financing settings.")
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    rf_series = load_rf_series(options.index.unique())
    strategy = build_vrp_strategy()
    scenarios = (
        Scenario(
            name="regt_no_financing",
            margin_model=RegTMarginModel(broad_index=False),
            margin_policy=MarginPolicy(apply_financing=False),
        ),
        Scenario(
            name="regt_series_financing",
            margin_model=RegTMarginModel(
                broad_index=False,
                house_multiplier=1.05,
            ),
            margin_policy=MarginPolicy(
                apply_financing=True,
                cash_rate_annual=rf_series,
                borrow_rate_annual=rf_series + 0.02,
                liquidation_mode="target",
            ),
        ),
    )

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        bt, rf_used, run_cfg = build_backtester(
            options=options,
            ticker=args.ticker,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_per_leg=args.commission_per_leg,
            hedge_fee_bps=args.hedge_fee_bps,
            margin_model=scenario.margin_model,
            margin_policy=scenario.margin_policy,
        )
        trades, mtm = bt.run()
        metrics = compute_performance_metrics(
            trades=trades,
            mtm_daily=mtm,
            risk_free_rate=rf_used,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "peak_initial_margin": f"{mtm['initial_margin_requirement'].max():.2f}",
                "financing_pnl": f"{mtm['financing_pnl'].sum():.2f}",
                "total_pnl": f"{metrics.trades.total_pnl:.2f}",
                "trade_count": str(metrics.trades.total_trades),
            }
        )

    print("Margin-model and financing comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"peak_initial_margin={row['peak_initial_margin']}, "
            f"financing_pnl={row['financing_pnl']}, "
            f"total_pnl={row['total_pnl']}, "
            f"trade_count={row['trade_count']}"
        )


if __name__ == "__main__":
    main()
