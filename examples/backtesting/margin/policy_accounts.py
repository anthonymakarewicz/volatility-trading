"""Compare account-level margin policies and liquidation behavior.

Run from repository root with:
`python -m examples.backtesting.margin.policy_accounts`
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.core.cli import parse_common_args
from examples.core.vrp_helpers import (
    build_backtester,
    build_vrp_strategy,
    load_options_window,
    load_rf_series,
)
from volatility_trading.backtesting import MarginPolicy


@dataclass(frozen=True)
class Scenario:
    """Margin-account policy used in one run."""

    name: str
    margin_policy: MarginPolicy


def main() -> None:
    args = parse_common_args("Compare margin-account liquidation policies.")
    options = load_options_window(ticker=args.ticker, start=args.start, end=args.end)
    rf_series = load_rf_series(options.index.unique())
    strategy = build_vrp_strategy(
        risk_budget_pct=2.0,
        margin_budget_pct=0.95,
    )
    scenarios = (
        Scenario(
            name="full_liquidation",
            margin_policy=MarginPolicy(
                apply_financing=True,
                cash_rate_annual=rf_series,
                borrow_rate_annual=rf_series + 0.02,
                liquidation_mode="full",
                margin_call_grace_days=1,
            ),
        ),
        Scenario(
            name="target_liquidation",
            margin_policy=MarginPolicy(
                apply_financing=True,
                cash_rate_annual=rf_series,
                borrow_rate_annual=rf_series + 0.02,
                liquidation_mode="target",
                liquidation_buffer_ratio=0.10,
                margin_call_grace_days=1,
            ),
        ),
    )

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        bt, _, _ = build_backtester(
            options=options,
            ticker=args.ticker,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_per_leg=args.commission_per_leg,
            hedge_fee_bps=args.hedge_fee_bps,
            margin_policy=scenario.margin_policy,
        )
        trades, mtm = bt.run()
        rows.append(
            {
                "scenario": scenario.name,
                "forced_liquidation_days": str(int(mtm["forced_liquidation"].sum())),
                "contracts_liquidated": str(int(mtm["contracts_liquidated"].sum())),
                "max_margin_call_days": str(int(mtm["margin_call_days"].max())),
                "financing_pnl": f"{mtm['financing_pnl'].sum():.2f}",
                "closed_trades": str(len(trades)),
            }
        )

    print("Margin-account policy comparison:")
    for row in rows:
        print(
            f"- {row['scenario']}: "
            f"forced_liquidation_days={row['forced_liquidation_days']}, "
            f"contracts_liquidated={row['contracts_liquidated']}, "
            f"max_margin_call_days={row['max_margin_call_days']}, "
            f"financing_pnl={row['financing_pnl']}, "
            f"closed_trades={row['closed_trades']}"
        )


if __name__ == "__main__":
    main()
