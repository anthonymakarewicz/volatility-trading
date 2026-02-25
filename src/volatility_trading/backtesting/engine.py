from .types import (
    BacktestConfig,
    DataMapping,
    ParamGrid,
    SliceContext,
    StrategyRunner,
)

# TODO: Fix circular import in init to put Backtester in init

# TODO: Pass to it the MarginAccount, or a Broker account, and move the execution logic here
# so as to avoid passign a Slice context to the strategy


class Backtester:
    def __init__(
        self,
        data: DataMapping,
        strategy: StrategyRunner,
        config: BacktestConfig,
        param_grid: ParamGrid | None = None,
    ):
        """
        data: Map of named DataFrames/Series, e.g.
            {
                "options": options_df,      # Option chain to execute trades
                "features": features_df,   # For signals and filters
                "hedge": hedge_series,    # for delta hedging
            }
        """
        self.data = data
        self.strategy = strategy
        self.config = config
        self.param_grid = param_grid or {}

    def run(self):
        current_capital = self.config.initial_capital

        ctx = SliceContext(
            data=self.data,
            params=self.param_grid,
            config=self.config,
            capital=current_capital,
        )

        trades, mtm = self.strategy.run(ctx)

        return trades, mtm

    # Should be a global orchestrator of the backtest
    # The startegy runs the stratgey bybreturnign tardes and mtm, and the backtester performs the optmization of
    # the params, runs the walk forwrad by passing the current train dev and test sets to the strategy
