from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    def __init__(self):
        self._z_score: pd.Series | None = None

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

    def get_z_score(self) -> pd.Series | None:
        """Return the most recent zscore,
        or None if not a zscore strategy."""
        return self._z_score
