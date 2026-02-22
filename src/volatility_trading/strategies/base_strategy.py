from abc import ABC, abstractmethod

from volatility_trading.backtesting.types import SliceContext
from volatility_trading.filters import Filter
from volatility_trading.signals import Signal

# TODO: Maybe make Signal as compulsory instead of optional here (list of filetrs keep optional)

# TODO: Myabe make SIgnal part of the Strategy like VRPHarvetsing only as ShortWalways, skew mispiricng
# only the z score.
# Or an enum of tolerated Signals for each stratgey, becasue skew can alos have a fixed threhsold instead of z score


class Strategy(ABC):
    def __init__(
        self, signal: Signal | None = None, filters: list[Filter] | None = None
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

        This fucntion will be used by the optmizer in Backtesting to find the best
        strategy parameters.
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
