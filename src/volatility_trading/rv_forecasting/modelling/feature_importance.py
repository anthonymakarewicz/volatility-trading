import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression


def single_feature_importance(X, y, model, cv):
    """
    Single-Feature Importance (SFI) using R².

    For each feature x_j:
      - Fit y ~ x_j in each CV train fold
      - Compute OOS R² on that fold's validation

    Returns
    -------
    sfi_df : DataFrame with mean/std R² per feature (sorted by mean_R2)
    scores : array of shape (n_folds, n_features) with raw per-fold R²
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    features = X.columns.to_list()

    n_features = X.shape[1]
    n_folds = cv.get_n_splits(X, y)
    scores = np.zeros((n_folds, n_features))

    for fold, (tr, val) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_val, y_val = X.iloc[val], y.iloc[val]

        for j, col in enumerate(features):
            x_tr = X_tr[[col]]  # 2D for sklearn
            x_val = X_val[[col]]

            model.fit(x_tr, y_tr)
            scores[fold, j] = model.score(x_val, y_val)  # R²

    sfi_df = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_R2": scores.mean(axis=0),
                "std_R2": scores.std(axis=0),
            }
        )
        .sort_values("mean_R2", ascending=False)
        .reset_index(drop=True)
    )

    return sfi_df, scores


def in_sample_stability(X, y, model, cv):
    """
    In-sample stability for a single model across CV folds.

    For linear models (Lasso / ElasticNet / OLS) → tracks `coef_`.
    For RandomForestRegressor → tracks `feature_importances_`.

    Returns
    -------
    values : ndarray, shape (n_folds, n_features)
    summary : DataFrame with columns ['feature', 'mean', 'std', 'mean_abs']
              sorted by |mean|.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    features = X.columns.to_list()

    n_features = X.shape[1]
    n_folds = cv.get_n_splits(X, y)
    values = np.zeros((n_folds, n_features))

    for fold, (tr, val) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]

        est = clone(model)
        est.fit(X_tr, y_tr)

        # --- pick attribute depending on model type ---
        if isinstance(est, (Lasso, ElasticNet, LinearRegression)):
            v = est.coef_
        elif isinstance(est, RandomForestRegressor):
            v = est.feature_importances_
        else:
            raise ValueError(
                f"Model type {type(est)} not supported in in_sample_stability."
            )

        values[fold, :] = np.asarray(v)

    mean_v = values.mean(axis=0)
    std_v = values.std(axis=0)

    summary = pd.DataFrame(
        {
            "feature": features,
            "mean": mean_v,
            "std": std_v,
            "mean_abs": np.abs(mean_v),
        }
    ).sort_values("mean_abs", ascending=False)

    return values, summary


def oos_perm_importance(
    X, y, model, cv, n_repeats=30, scoring="neg_mean_squared_error", random_state=0
):
    """
    Out-of-sample permutation importance across CV folds.

    Returns
    -------
    pi : ndarray, shape (n_folds, n_features)
        Permutation importance per fold.
    summary : DataFrame
        Columns: ['feature', 'mean', 'std', 'mean_abs'],
        sorted by |mean| descending.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    features = X.columns.to_list()

    n_features = X.shape[1]
    n_folds = cv.get_n_splits(X, y)

    pi = np.zeros((n_folds, n_features))

    for fold, (tr, val) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_val, y_val = X.iloc[val], y.iloc[val]

        model.fit(X_tr, y_tr)

        r = permutation_importance(
            model,
            X_val,
            y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        pi[fold, :] = r.importances_mean

    # summary
    mean_v = pi.mean(axis=0)
    std_v = pi.std(axis=0)

    summary = pd.DataFrame(
        {
            "feature": features,
            "mean": mean_v,
            "std": std_v,
            "mean_abs": np.abs(mean_v),
        }
    ).sort_values("mean_abs", ascending=False)

    return pi, summary
