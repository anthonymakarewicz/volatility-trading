import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def qlike(y_true, y_pred):
    """
    QLIKE on variance scale.
    Here y_true / y_pred are log-variance -> exp back to variance.
    """
    var_true = np.exp(y_true)
    var_pred = np.exp(y_pred)
    return np.mean(var_pred / var_true - np.log(var_pred / var_true) - 1.0)


def r2_oos(y_true, y_pred, y_pred_bench):
    """
    R^2_OOS(model | bench) = 1 - SSE_model / SSE_bench
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_bench = np.asarray(y_pred_bench)

    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - y_pred_bench)**2)
    return 1.0 - num / den


def compute_metrics(y_true, y_pred, y_pred_bench=None):
    """
    Convenience helper: standard R², MSE, QLIKE.
    """
    import pandas as pd
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    res = y_true - y_pred

    out = {
        "R2":      r2_score(y_true, y_pred),
        "MSE":     mean_squared_error(y_true, y_pred),
        "QLIKE":   qlike(y_true, y_pred),
        "Var_res": float(np.var(res, ddof=1)),
    }

    if y_pred_bench is not None:
        out["R2_oos"] = r2_oos(y_true, y_pred, y_pred_bench)
    else:
        out["R2_oos"] = 0.0  # for the benchmark itself

    return out


def compute_subperiod_metrics(perf, subperiods):
    """
    Compute metrics for HAR-RV-VIX and Naive IV on each sub-period,
    always using Naive RV as the benchmark in R2_oos.

    Parameters
    ----------
    perf : DataFrame
        Must contain columns: ['y_true', 'har_vix', 'naive_iv', 'naive_rv']
    subperiods : list of (start, end, label)
        e.g. [("2010-01-01", "2012-12-31", "2010–2012"), ...]

    Returns
    -------
    DataFrame indexed by [period, model] with all metrics
    returned by `compute_metrics` (R2, MSE, QLIKE, Var_res, R2_oos, ...).
    """
    import pandas as pd
    
    rows = []

    for start, end, label in subperiods:
        mask = (perf.index >= start) & (perf.index <= end)
        df_sub = perf.loc[mask]

        if df_sub.empty:
            continue

        y_true_sub   = df_sub["y_true"]
        y_har_vix_sub = df_sub["har_vix"]
        y_iv_sub     = df_sub["naive_iv"]
        y_rv_sub     = df_sub["naive_rv"]

        # metrics relative to Naive RV
        m_har_vix = compute_metrics(
            y_true_sub, 
            y_har_vix_sub, 
            y_pred_bench=y_rv_sub
        )
        m_iv = compute_metrics(
            y_true_sub, 
            y_iv_sub,
            y_pred_bench=y_rv_sub
        )

        row_har = {"period": label, "model": "HAR-RV-VIX"}
        row_har.update(m_har_vix)
        rows.append(row_har)

        row_iv = {"period": label, "model": "Naive_IV"}
        row_iv.update(m_iv)
        rows.append(row_iv)

    metrics_df = pd.DataFrame(rows).set_index(["period", "model"])
    return metrics_df