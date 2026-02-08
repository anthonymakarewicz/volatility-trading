from .base_filter import Filter


class VIXFilter(Filter):
    def __init__(self, panic_threshold=25, mom_threshold=0.10):
        self.panic_threshold = panic_threshold
        self.mom_threshold = mom_threshold

    def get_params(self):
        return {
            "vixfilter__panic_threshold": self.panic_threshold,
            "vixfilter__mom_threshold": self.mom_threshold,
        }

    def set_params(self, panic_threshold=None, mom_threshold=None, **kwargs):
        if panic_threshold is not None:
            self.panic_threshold = panic_threshold
        if mom_threshold is not None:
            self.mom_threshold = mom_threshold
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected parameters passed to VIXFilter: {unexpected}")

    def apply(self, signals, ctx):
        vix = ctx["vix"]
        vix_mask = vix < self.panic_threshold
        out = signals.copy()
        out.loc[~vix_mask, ["long", "short", "exit"]] = False
        return out
