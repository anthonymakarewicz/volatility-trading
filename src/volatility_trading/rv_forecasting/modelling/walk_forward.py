import pandas as pd
import numpy as np
from sklearn.base import clone

class WalkForwardOOS:
    """
    Generic walk-forward backtester for time series regressors.

    - model_agnostic: any sklearn estimator / Pipeline with fit/predict
    - can do rolling or expanding windows
    - builds rebalancing dates from the feature index
    """
    def __init__(
        self,
        estimator,
        start_backtest,
        end_backtest,
        window_years=5,
        expanding=False,
        rebal_freq="ME", # "ME" month-end, "MS" month-start, "W" weekly
        purge_horizon=0, # days to purge before test window,
        min_train_samples=100,
    ):
        self.estimator = estimator
        self.start_backtest = pd.Timestamp(start_backtest)
        self.end_backtest = pd.Timestamp(end_backtest)
        self.window_years = window_years
        self.expanding = expanding
        self.rebal_freq = rebal_freq
        self.purge_horizon = purge_horizon
        self.min_train_samples = min_train_samples

    def _make_rebal_dates(self, index):
        """
        Map calendar rebal dates to actual available index dates.
        """
        raw_rebals = pd.date_range(
            self.start_backtest,
            self.end_backtest,
            freq=self.rebal_freq,
        )
        
        idx = index.sort_values()
        rebal_dates = []
        for d in raw_rebals:
            if d < idx[0]:
                continue
            pos = idx.searchsorted(d, side="right") - 1
            if pos >= 0:
                rebal_dates.append(idx[pos])

        return sorted(np.unique(rebal_dates))

    def run(self, X, y):
        """
        Run walk-forward and return a DataFrame with y_true, y_pred.
        X, y must share the same DateTimeIndex.
        """
        X = X.sort_index()
        y = y.sort_index()

        idx = X.index.intersection(y.index)
        X = X.loc[idx]
        y = y.loc[idx]

        rebal_dates = self._make_rebal_dates(X.index)
        pred_frames = []

        for i, t_rebal in enumerate(rebal_dates):
            purge_end = t_rebal - pd.Timedelta(days=self.purge_horizon)

            # ----- train window -----
            if self.expanding:
                train_mask = (X.index < purge_end)
            else:
                train_start = t_rebal - pd.DateOffset(years=self.window_years)
                train_mask = (X.index >= train_start) & (X.index < purge_end)

            X_tr = X.loc[train_mask]
            y_tr = y.loc[train_mask]

            if len(X_tr) < self.min_train_samples:
                continue

            est = clone(self.estimator) # get a fresh model
            est.fit(X_tr, y_tr)

            # ----- test window -----
            if i < len(rebal_dates) - 1:
                t_next = rebal_dates[i + 1]
            else:
                t_next = self.end_backtest + pd.Timedelta(days=1)

            test_mask = (
                (X.index >= t_rebal) &
                (X.index < t_next) &
                (X.index >= self.start_backtest) &
                (X.index <= self.end_backtest)
            )

            X_te = X.loc[test_mask]
            y_te = y.loc[test_mask]

            if X_te.empty:
                continue

            y_hat = est.predict(X_te)

            pred_frames.append(
                pd.DataFrame(
                    {"y_true": y_te, "y_pred": y_hat},
                    index=y_te.index,
                )
            )

        if not pred_frames:
            return None

        return pd.concat(pred_frames).sort_index()