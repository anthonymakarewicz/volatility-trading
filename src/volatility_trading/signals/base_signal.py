from abc import ABC, abstractmethod

import pandas as pd

# TODO: Fix the z-score attribute


class Signal(ABC):
    def __init__(self):
        self._z_score = None

    @abstractmethod
    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Return signal flags with boolean columns: `long`, `short`, `exit`."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return current hyper-parameters as {name: value}."""
        ...

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Update internal hyper-parameters."""
        ...

    def get_z_score(self):
        """Return the most recent zscore,
        or None if not a zscore strategy."""
        return self._z_score
