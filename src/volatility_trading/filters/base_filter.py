from abc import ABC, abstractmethod


class Filter(ABC):
    @abstractmethod
    def apply(self, signals, ctx):
        pass  # returns the filtered signals

    @abstractmethod
    def get_params(self):
        """Return current hyper-parameters as {name: value}."""
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        """Update internal hyper-parameters."""
        pass
