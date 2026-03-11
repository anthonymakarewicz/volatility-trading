"""Private signal wrappers used to compose higher-level strategy behavior."""

from __future__ import annotations

import pandas as pd

from .base_signal import Signal


class InvertedSignal(Signal):
    """Swap `long` and `short` entries from another signal.

    The wrapped signal remains the source of parameter state and any auxiliary
    values such as z-scores. Exit flags are preserved unchanged.
    """

    def __init__(self, base_signal: Signal) -> None:
        super().__init__()
        self.base_signal = base_signal

    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Return wrapped signals with entry directions inverted."""
        signals = self.base_signal.generate_signals(data).copy()
        long_entries = signals["long"].copy()
        signals["long"] = signals["short"]
        signals["short"] = long_entries
        self._z_score = self.base_signal.get_z_score()
        return signals

    def get_params(self) -> dict:
        """Return parameters from the wrapped signal."""
        return self.base_signal.get_params()

    def set_params(self, **kwargs) -> None:
        """Forward parameter updates to the wrapped signal."""
        self.base_signal.set_params(**kwargs)
        self._z_score = self.base_signal.get_z_score()
