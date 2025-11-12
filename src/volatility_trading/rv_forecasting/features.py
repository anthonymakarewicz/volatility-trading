import numpy as np
import pandas as pd
from .vol_estimators import rv_intraday


def create_forward_target(daily_variance: pd.Series, horizon: int = 21) -> pd.Series:
    """
    daily_variance: per-day realized variance (NOT sqrt, NOT annualized).
                    index = trading days.
    horizon: forecast horizon in trading days (e.g. 21 ~ 1 month).

    Returns:
        y_t = log( average future variance from t+1 to t+horizon )
    """
    # trailing mean, then realign to t as forward mean
    fwd_avg_var = (
        daily_variance
        .rolling(horizon).mean()       # at t+h: mean of [t+1 .. t+h]
        .shift(-horizon)               # move it back so it's aligned at t
    )

    y = np.log(fwd_avg_var)
    y.name = f"log_fwd_var_{horizon}d"
    return y


def create_har_lags(real_variance):
    X_har = pd.DataFrame({
        "RV_D": real_variance,                   # yesterday’s daily RV
        "RV_W": real_variance.rolling(5).mean(), # weekly avg of daily RVs
        "RV_M": real_variance.rolling(21).mean() # monthly avg of daily RVs
    })
    return X_har



def build_har_vix_dataset(es_5min, start=None, end=None, h=21):
    # 1) daily RV
    daily_rv = rv_intraday(es_5min["close"])

    if start is not None:
        daily_rv = daily_rv.loc[start:]
    if end is not None:
        daily_rv = daily_rv.loc[:end]

    # 2) target
    y = create_forward_target(daily_rv, horizon=h)

    # 3) predictors
    X_har = create_har_lags(daily_rv)   # RV_D, RV_W, RV_M, ...

    # use the target index to get the proper VIX range
    feat_start = y.index.min()
    feat_end   = y.index.max()

    X_vix = create_market_features(start=feat_start, end=feat_end)[["VIX"]]
    X_vix = X_vix.reindex(y.index).ffill()

    X = pd.concat([X_har, X_vix], axis=1)

    # 4) align and drop NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X_final = data[X.columns]
    y_final = data[y.name]

    return X_final, y_final
