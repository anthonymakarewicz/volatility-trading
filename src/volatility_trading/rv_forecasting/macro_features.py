import os
import pandas as pd
from dotenv import load_dotenv

try:
    from fredapi import Fred
except ImportError:
    Fred = None

_FRED = None


def _get_fred_client():
    global _FRED
    if _FRED is not None:
        return _FRED
    if Fred is None:
        return None

    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None

    _FRED = Fred(api_key=api_key)
    return _FRED


def _fred_series(series_id, start=None, end=None) -> pd.Series:
    fred = _get_fred_client()
    if fred is None:
        return pd.Series(dtype=float, name=series_id)

    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = pd.Series(s, name=series_id)
    s.index = pd.to_datetime(s.index)
    return s


def create_macro_features(start="2005-01-01", end=None) -> pd.DataFrame:
    DGS10  = _fred_series("DGS10",  start, end).rename("DGS10")
    DGS2   = _fred_series("DGS2",   start, end).rename("DGS2")
    DGS3MO = _fred_series("DGS3MO", start, end).rename("DGS3MO")
    term_spread = (DGS10 - DGS3MO).rename("term_spread_10y_3m")

    HY_OAS = _fred_series("BAMLH0A0HYM2", start, end).rename("HY_OAS")
    IG_OAS = _fred_series("BAMLC0A0CM",   start, end).rename("IG_OAS")

    df = pd.concat([DGS10, DGS2, DGS3MO, term_spread, HY_OAS, IG_OAS], axis=1).sort_index()
    if df.empty:
        return df

    return df.asfreq("B").ffill()