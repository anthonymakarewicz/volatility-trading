import pandas as pd
from .base_signal import Signal

# TODO: Shift the rolling mean and std for z score computation

class ZScore(Signal):
    def __init__(self, window=60, entry=2.0, exit=0.5):
        self.window = window
        self.entry = entry
        self.exit  = exit

    def get_params(self):
        return {
            "strategy__window": self.window,
            "strategy__entry":  self.entry,
            "strategy__exit":   self.exit,
        }

    def set_params(self, window=None, entry=None, exit=None, **kwargs):
        if window is not None: self.window = window
        if entry is not None:  self.entry  = entry
        if exit is not None:   self.exit   = exit
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected parameters passed to ZScoreStrategy: {unexpected}")

    def generate_signals(self, skew_series):
        z_score = compute_zscore(skew_series, window=self.window)
        signals = pd.DataFrame(index=z_score.index)
        signals["long"] = False
        signals["short"] = False
        signals["exit"] = False

        self._z_score = z_score

        position = 0  # +1 = long, -1 = short, 0 = flat
        for i in range(len(z_score)):
            if pd.isna(z_score.iloc[i]):  # Skip NaN values for rolling
                continue

            if position == 0: 
                # Enter long position
                if z_score.iloc[i] < -self.entry:
                    signals.at[z_score.index[i], "long"] = True
                    position = 1
                # Enter short position
                elif z_score.iloc[i] > self.entry:
                    signals.at[z_score.index[i], "short"] = True
                    position = -1 

            elif position == 1: 
                # Exit long position
                if z_score.iloc[i] >= -self.exit:
                    signals.at[z_score.index[i], "exit"] = True
                    position = 0

            elif position == -1:
                # Exit short position
                if z_score.iloc[i] <= self.exit:
                    signals.at[z_score.index[i], "exit"] = True
                    position = 0

        return signals  
    

def compute_zscore(series, window=60):
    return (
        (series - series.rolling(window).mean()) /
         series.rolling(window).std()
    )