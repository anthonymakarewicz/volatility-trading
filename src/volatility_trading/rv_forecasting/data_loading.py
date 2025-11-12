import pandas as pd

def load_intraday_prices(path, start=None, end=None):
    es_5min = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date","time","open","high","low","close","volume"],
    )
    es_5min["datetime"] = pd.to_datetime(
        es_5min["date"] + " " + es_5min["time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    es_5min = (
        es_5min
        .drop(columns=["date","time"])
        .set_index("datetime")
        .sort_index()
    )
    if start is not None:
        es_5min = es_5min.loc[start:]
    if end is not None:
        es_5min = es_5min.loc[:end]

    es_5min = es_5min.dropna(subset=["close"], axis=0)
    es_5min = es_5min[~es_5min.index.duplicated(keep="last")]
    return es_5min