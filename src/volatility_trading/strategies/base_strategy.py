from abc import ABC, abstractmethod
from typing import List

from volatility_trading.signals import Signal
from volatility_trading.filters import Filter
from volatility_trading.backtesting.types import SliceContext 


class Strategy(ABC):
    def __init__(
        self, 
        signal: Signal | None = None, 
        filters: List[Filter] | None = None
    ):
        self.signal = signal
        self.filters = filters or []

    @abstractmethod
    def run(self, ctx: SliceContext):
        """Run strategy on a given SliceContext and return (trades, mtm)."""
        pass

    def set_params(self, flat_params: dict):
        """
        Split flat params into:
        - strategy__*   -> self.*
        - signal__*     -> self.signal.set_params(...)
        - <filter>__*   -> filt.set_params(...)
        """
        # Strategy-level params
        strat_params = {
            k.split("__", 1)[1]: v
            for k, v in flat_params.items()
            if k.startswith("strategy__")
        }
        for k, v in strat_params.items():
            setattr(self, k, v)

        # Signal-level params
        if self.signal is not None:
            sig_params = {
                k.split("__", 1)[1]: v
                for k, v in flat_params.items()
                if k.startswith("signal__")
            }
            if sig_params:
                self.signal.set_params(**sig_params)

        # Filter-level params
        for filt in self.filters:
            prefix = filt.__class__.__name__.lower() + "__"
            f_params = {
                k.split("__", 1)[1]: v
                for k, v in flat_params.items()
                if k.startswith(prefix)
            }
            if f_params:
                filt.set_params(**f_params)