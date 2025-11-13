import numpy as np
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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