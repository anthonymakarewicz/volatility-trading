import pandas as pd

from .base_signal import Signal


class LongOnlySignal(Signal):
    def get_params(self) -> dict:
        # No hyperparameters to tune
        return {}

    def set_params(self, **kwargs) -> None:
        # Nothing to set
        pass

    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["long"] = True
        signals["short"] = False
        signals["exit"] = False
        return signals


class ShortOnlySignal(Signal):
    def get_params(self) -> dict:
        return {}

    def set_params(self, **kwargs) -> None:
        pass

    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["long"] = False
        signals["short"] = True
        signals["exit"] = False
        return signals
