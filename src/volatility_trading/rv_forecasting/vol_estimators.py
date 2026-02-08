import numpy as np
import pandas as pd


def rv_close_to_close(returns, h=21, ann=252):
    var = returns.pow(2).rolling(h).sum() * (ann / h)
    return np.sqrt(var)


def rv_parkinson(high, low, h=21, ann=252):
    rp = np.log(high / low).pow(2)  # per-day range variance proxy
    var = rp.rolling(h, min_periods=h).sum() * (ann / (4 * np.log(2) * h))
    return np.sqrt(var)


def rv_garman_klass(open_, high, low, close, h=21, ann=252):
    # Daily GK variance (no drift)
    rs = 0.5 * (np.log(high / low).pow(2)) - (2 * np.log(2) - 1) * (
        np.log(close / open_).pow(2)
    )
    var = rs.rolling(h).sum() * (ann / h)
    return np.sqrt(var)


def rv_rogers_satchell(open_, high, low, close, h=21, ann=252):
    term1 = np.log(high / close) * np.log(high / open_)
    term2 = np.log(low / close) * np.log(low / open_)
    rs = term1 + term2
    var = rs.rolling(h).sum() * (ann / h)
    return np.sqrt(var)


def rv_yang_zhang(open_, high, low, close, h=21, ann=252, k=0.34):
    # Overnight return variance
    oc = np.log(open_ / close.shift(1))
    sigma_o = oc.pow(2)

    # Open-to-close variance
    co = np.log(close / open_)
    sigma_c = co.pow(2)

    # Rogers–Satchell component
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(
        low / open_
    )

    # Daily YZ variance
    yz = sigma_o + k * sigma_c + (1 - k) * rs

    var = yz.rolling(h, min_periods=h).sum() * (ann / h)
    return np.sqrt(var)


def rv_intraday(close, rth_start: str = "09:30", rth_end: str = "16:00"):
    """
    Compute daily realized variance from intraday data:
        RV_t = sum_i r_{t,i}^2, where r_{t,i} = log(P_{t,i}/P_{t,i-1})
    """
    # intraday log returns
    tod = close.index.strftime("%H:%M")
    rth = close[(tod >= rth_start) & (tod <= rth_end)].copy()
    r = np.log(rth).diff()

    # sum of squared intraday returns by day
    rv_daily = r.pow(2).groupby(rth.index.date).sum()
    rv_daily.index = pd.to_datetime(rv_daily.index)
    rv_daily.name = "var_daily"

    return rv_daily
