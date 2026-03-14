"""End-to-end skew mispricing backtest example on processed datasets.

This script demonstrates a minimal pipeline:
1) load processed options + daily features + rates datasets,
2) build a skew-mispricing strategy spec,
3) run the backtest,
4) print performance metrics.

Run from repository root with:
`python -m examples.backtesting.strategies.skew_mispricing_end_to_end`
"""

from __future__ import annotations

from examples.backtesting.strategies import build_skew_strategy
from examples.core.backtesting_helpers import (
    build_backtester,
    load_options_window,
    run_and_report,
)
from examples.core.cli import parse_common_args
from volatility_trading.backtesting import load_daily_features_frame


def main() -> None:
    cfg = parse_common_args("Run a minimal skew-mispricing E2E backtest.")
    options = load_options_window(ticker=cfg.ticker, start=cfg.start, end=cfg.end)
    features = load_daily_features_frame(
        ticker=cfg.ticker,
        start=cfg.start,
        end=cfg.end,
    )

    strategy = build_skew_strategy()
    bt, rf_series, _ = build_backtester(
        options=options,
        ticker=cfg.ticker,
        strategy=strategy,
        features=features,
        initial_capital=cfg.initial_capital,
        commission_per_leg=cfg.commission_per_leg,
        hedge_fee_bps=cfg.hedge_fee_bps,
        broad_index=cfg.ticker in {"SPX", "SPXW"},
    )
    run_and_report(
        backtester=bt,
        initial_capital=cfg.initial_capital,
        rf_series=rf_series,
    )


if __name__ == "__main__":
    main()
