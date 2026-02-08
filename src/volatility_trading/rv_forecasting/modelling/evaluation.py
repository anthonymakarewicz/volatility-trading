import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from .data_processing import DataProcessor
from .metrics import compute_metrics


def eval_model_cv(
    name: str,
    base_estimator,
    features,
    X,
    y,
    cv,
    dp_kwargs,
    y_pred_bench=None,
    n_jobs: int = -1,
):
    """
    Run CV for a model, compute metrics, optionally R2_oos vs a benchmark,
    and return (metrics_dict, y_pred_series).
    """
    # subset features
    X_sub = X[features].copy()

    # pipeline: DP + model
    pipe = Pipeline(
        [
            ("dp", DataProcessor(**dp_kwargs)),
            ("model", base_estimator),
        ]
    )

    # CV predictions
    y_pred = cross_val_predict(
        pipe,
        X_sub,
        y,
        cv=cv,
        n_jobs=n_jobs,
    )
    y_pred = pd.Series(y_pred, index=y.index)

    # metrics
    metrics = compute_metrics(y, y_pred, y_pred_bench)
    metrics["model"] = name

    return metrics, y_pred


def eval_ensembles(
    y,
    y_pred_a,
    y_pred_b,
    weights,
    y_pred_bench=None,
    label_a="Model_A",
    label_b="Model_B",
):
    """
    Evaluate linear ensembles of two prediction series.

    Parameters
    ----------
    y : pd.Series
        True target.
    y_pred_a : pd.Series
        Predictions from first model.
    y_pred_b : pd.Series
        Predictions from second model.
    weights : list[float]
        Weights on model A, e.g. [0.5, 0.7, 0.8].
    y_pred_bench : pd.Series or None
        Benchmark predictions for R2_oos. If None, R2_oos = 0.
    label_a, label_b : str
        Names used in the model label.

    Returns
    -------
    df_metrics : DataFrame
        One row per ensemble with metrics.
    ens_preds : dict[float -> pd.Series]
        Map weight -> ensemble prediction series.
    """
    # align on common index
    y, y_pred_a = y.align(y_pred_a, join="inner")
    y, y_pred_b = y.align(y_pred_b, join="inner")

    rows = []
    ens_preds = {}

    for w in weights:
        y_hat_ens = w * y_pred_a + (1.0 - w) * y_pred_b
        ens_preds[w] = y_hat_ens

        m = compute_metrics(y, y_hat_ens, y_pred_bench)
        m["model"] = f"{w:.2f} * {label_a} + {1 - w:.2f} * {label_b}"
        rows.append(m)

    df_metrics = pd.DataFrame(rows).set_index("model")
    return df_metrics, ens_preds
