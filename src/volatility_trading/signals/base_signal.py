from abc import ABC, abstractmethod

# TODO: Fix the z-score attribute


class Signal(ABC):
    def __init__(self):
        self._z_score = None

    @abstractmethod
    def generate_signals(self, skew_series):
        """Returns a DataFrame of boolean columns [long, short, exit]"""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return current hyper-parameters as {name: value}."""
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        """Update internal hyper-parameters."""
        pass

    def get_z_score(self):
        """Return the most recent zscore,
        or None if not a zscore strategy."""
        return self._z_score
