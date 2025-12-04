@dataclass
class SliceContext:
    #data: dict[str, pd.DataFrame]
    data: Mapping[str, Any]   # {"options": df, "features": df, "hedge": series, ...}
    params: dict
    config: BacktestConfig
    capital: float # current capital


class VRPStrategy(Strategy):
    def __init__(
        self,
        signal: Signal,
        filters: list[Filter] | None = None,
        target_dte: int = 30,
        holding_period: int = 5,
        stop_loss_pct: float = 0.8,
        take_profit_pct: float = 0.5,
    ):
        """
        Baseline VRP harvesting strategy:
        - short ATM straddle when signal is ON
        - 30D target maturity (by dte)
        - simple SL/TP in units of a stress-based risk per contract
        - no delta-hedge (for now)
        """
        super().__init__(signal=signal, filters=filters)
        self.target_dte = target_dte
        self.holding_period = holding_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct


class Backtester:
    def __init__(
        self, 
        data: Mapping[str, Any], 
        strategy: Strategy, 
        config: BacktestConfig,
        param_grid: dict | None = None

    ):
        """
        data: dict of named DataFrames/Series, e.g.
            {
                "options": options_df,      # Option chain to execute trades
                "features": features_df,   # For signals and filters
                "hedge": hedge_series,    # for delta hedging
            }
        """
        self.data = data
        self.strategy = strategy
        self.config = config
        self.param_grid = param_grid

    def _rolling_windows(self):
        # your existing logic
        ...

    def _slice_data(self, start, end):
        sliced = {}
        for name, df in self.data.items():
            sliced[name] = df.loc[start:end]
        return sliced

    def run(self):
        results = []
        current_capital = self.config.initial_capital

        for train_idx, val_idx, test_idx in self._rolling_windows(dates):
            train_idx = window.train_idx
            val_idx   = window.val_idx
            test_idx  = window.test_idx

            # --- 1) Optimise params on train (+ optional val) ---
            if self.param_grid is not None:
                best_params = self._find_best_params(train_idx, val_idx)
                if best_params is not None:
                    self.strategy.set_params(best_params)
            else:
                best_params = None

            # --- 2) Build test SliceContext and run strategy ---
            test_data = self._slice_data(test_idx)
            ctx_test = SliceContext(
                data=test_data,
                params=best_params or {},
                config=self.config,
                capital=current_capital,
            )

            trades, mtm = self.strategy.run_slice(ctx_test)

            # --- 3) Update capital, store results ---
            if mtm is not None and not mtm.empty:
                current_capital = mtm["equity"].iloc[-1]

            results.append(
                {
                    "window": window,
                    "params": best_params,
                    "trades": trades,
                    "mtm": mtm,
                    "end_capital": current_capital,
                }
            )

        return results