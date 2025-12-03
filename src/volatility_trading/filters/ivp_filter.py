from .base_filter import Filter


class IVPFilter(Filter):
    def __init__(self, window=252, lower=30, upper=70):
        self.window = window
        self.lower = lower / 100
        self.upper = upper / 100

    def get_params(self):
        return {
            "ivpfilter__window": self.window,
            "ivpfilter__lower":  int(self.lower * 100),
            "ivpfilter__upper":  int(self.upper * 100),
        }

    def set_params(self, window=None, lower=None, upper=None, **kwargs):
        if window is not None:
            self.window = window
        if lower is not None:
            self.lower = lower / 100
        if upper is not None:
            self.upper = upper / 100
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected parameters passed to IVPFilter: {unexpected}")

    def apply(self, signals, ctx):

        if "iv_atm" not in ctx:
            raise ValueError("'iv_atm' column is missing in context dataframe")
        # or maybe pass the column name as argument to the class 
        
        iv_atm = ctx["iv_atm"]
        ivp = compute_iv_percentile(
            iv_atm, 
            window=self.window
        )

        out = signals.copy()
        ivp_mask = (
            (ivp >= self.lower) &
            (ivp <= self.upper) 
        )
        out.loc[~ivp_mask, ['long','short','exit']] = False

        return out


def compute_iv_percentile(iv_series, window=252):
    return iv_series.rolling(window).apply(
        lambda x: (x[-1] > x).sum() / window,
        raw=True
    )