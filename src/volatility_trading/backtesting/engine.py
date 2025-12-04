from .types import BacktestConfig, SliceContext, DataMapping, ParamGrid
from volatility_trading.strategies import Strategy

class Backtester:
    def __init__(
        self, 
        data: DataMapping, 
        strategy: Strategy, 
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
            params= self.param_grid,
            config=self.config,
            capital=current_capital,
        )

        trades, mtm = self.strategy.run(ctx)

        return trades, mtm