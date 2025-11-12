import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_transform_demo(series, use_log=False, use_sqrt=False, winsorize=None):
    raw = series.dropna().to_numpy()

    transformed = raw.copy()
    labels = []

    # Winsorization
    if winsorize is not None:
        low_q, high_q = winsorize
        q_low, q_high = np.quantile(transformed, [low_q, high_q])
        transformed = np.clip(transformed, q_low, q_high)
        labels.append(f"Winsor[{low_q:.3f},{high_q:.3f}]")

    # Sqrt transform
    if use_sqrt:
        transformed = np.sqrt(np.clip(transformed, 0.0, None))
        labels.append("sqrt")

    # Log transform
    if use_log:
        transformed = np.log(np.clip(transformed, 1e-8, None))
        labels.append("log")

    tlabel = " + ".join(labels) if labels else "raw"

    # stats
    stats_raw = (skew(raw), kurtosis(raw))
    stats_trans = (skew(transformed), kurtosis(transformed))

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # raw
    axes[0].hist(raw, bins=40, alpha=0.7, color="steelblue")
    axes[0].set_title(f"Distribution of {series.name} (raw)")
    axes[0].text(
        0.95, 0.95,
        f"Skew={stats_raw[0]:.2f}\nKurt={stats_raw[1]:.2f}",
        transform=axes[0].transAxes, ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # transformed
    axes[1].hist(transformed, bins=40, alpha=0.7, color="darkorange")
    axes[1].set_title(f"Distribution of {series.name} ({tlabel})")
    axes[1].text(
        0.95, 0.95,
        f"Skew={stats_trans[0]:.2f}\nKurt={stats_trans[1]:.2f}",
        transform=axes[1].transAxes, ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def plot_mean_std_importance(
    ax, values, feature_names, title, top_n=20, abs_values=False
):
    """
    Plot mean ± std importance on a given axis.
    values: (n_folds, n_features)
    """
    vals = np.array(values)
    if abs_values:
        vals = np.abs(vals)

    mean_imp = vals.mean(axis=0)
    std_imp = vals.std(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean": mean_imp,
        "std": std_imp,
    })

    df = df.reindex(df["mean"].abs().sort_values(ascending=False).index)
    df_top = df.head(top_n)

    ax.barh(df_top["feature"], df_top["mean"], xerr=df_top["std"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean importance" + (" (|value|)" if abs_values else ""))
    ax.set_title(title)


def plot_purged_kfold_splits(cv, X, y):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    dates = X.index

    plt.figure(figsize=(10, 3))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        y_level = fold
        plt.scatter(dates[tr], np.full_like(tr, y_level),
                    marker='s', s=4, label='train' if fold == 0 else None, c="blue")
        plt.scatter(dates[val], np.full_like(val, y_level),
                    marker='s', s=10, label='val' if fold == 0 else None, c="orange")

    plt.yticks(range(cv.get_n_splits()), [f"Fold {k}" for k in range(cv.get_n_splits())])
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Date")
    plt.title("Purged K-Fold scheme (train vs validation over time)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_lasso_coef_paths(coefs, feature_names, top_n=10):
    """
    Plot evolution of top-n Lasso coefficients across CV folds.

    Parameters
    ----------
    coefs : array-like, shape (n_folds, n_features)
        Lasso coefficients per fold.
    feature_names : array-like, length n_features
    top_n : int
        Number of features to display (by mean |coef|).
    """
    coefs = np.asarray(coefs)
    feature_names = np.asarray(feature_names)
    n_folds, n_features = coefs.shape

    # rank features by mean |coef|
    mean_abs = np.abs(coefs).mean(axis=0)
    idx = np.argsort(-mean_abs)[:top_n]

    coefs_top = coefs[:, idx]
    feats_top = feature_names[idx]

    folds = np.arange(n_folds)

    plt.figure(figsize=(10, 5))
    for j, feat in enumerate(feats_top):
        plt.plot(folds, coefs_top[:, j], marker="o", label=feat)

    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("CV fold")
    plt.ylabel("Coefficient value")
    plt.title(f"Lasso coefficient paths across folds (top {top_n} by |β|)")
    plt.xticks(folds, [f"{k}" for k in folds])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_cv_mse_comparison(scores_a, scores_b, label_a="Model A", label_b="Model B"):
    """
    scores_a, scores_b: cross_val_score outputs with scoring="neg_mean_squared_error"
                        (one value per fold, negative MSE)
    label_a, label_b: labels for the two specs
    """
    # convert to positive MSE
    mse_a = -np.asarray(scores_a)
    mse_b = -np.asarray(scores_b)
    diff = mse_b - mse_a  # <0 → model B better

    fold_idx = np.arange(len(mse_a))

    print(f"Folds where {label_b} better:", (diff < 0).sum(), "/", len(diff))
    print("Mean ΔMSE (B - A):", diff.mean())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left: per-fold MSE
    axes[0].plot(fold_idx, mse_a, "o-", label=label_a)
    axes[0].plot(fold_idx, mse_b, "o-", label=label_b)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Per-fold MSE")
    axes[0].legend()

    # right: difference
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].bar(fold_idx, diff)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("ΔMSE (B - A)")
    axes[1].set_title("MSE difference per fold")

    plt.tight_layout()
    plt.show()


def plot_model_comparison_ts(y, y_pred_1, y_pred_2,
                                     label_1,
                                     label_2):
    # --- 1) Time-series overlay ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index, y.values, label="True RV")
    ax.plot(y_pred_1.index, y_pred_1.values, label=label_1, alpha=0.8)
    ax.plot(y_pred_2.index, y_pred_2.values, label=label_2, alpha=0.8)
    ax.set_title("True vs predicted 21D RV")
    ax.set_xlabel("Date")
    ax.set_ylabel("RV")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison_scatter(y, y_pred_1, y_pred_2,
                                label_1,
                                label_2,):
    # align everything just in case
    y, y_pred_1 = y.align(y_pred_1, join="inner")
    y, y_pred_2 = y.align(y_pred_2, join="inner")
    y_pred_1, y_pred_2 = y_pred_1.align(y_pred_2, join="inner")

    # common limits for true vs pred
    vals_true_lin_rf = np.concatenate([y.values, y_pred_1.values, y_pred_2.values])
    vmin = np.nanmin(vals_true_lin_rf)
    vmax = np.nanmax(vals_true_lin_rf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

    # 1) True vs linear
    axes[0].scatter(y, y_pred_1, s=10, alpha=0.6)
    axes[0].plot([vmin, vmax], [vmin, vmax], lw=1)
    axes[0].set_title(label_1)
    axes[0].set_xlabel("True RV")
    axes[0].set_ylabel("Predicted RV")

    # 2) True vs RF
    axes[1].scatter(y, y_pred_2, s=10, alpha=0.6)
    axes[1].plot([vmin, vmax], [vmin, vmax], lw=1)
    axes[1].set_title(label_2)
    axes[1].set_xlabel("True RV")
    axes[1].set_ylabel("Predicted RV")

    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series, lags=40, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF {title}")

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")  # Yule-Walker-M estimator
    axes[1].set_title(f"PACF {title}")

    plt.tight_layout()
    plt.show()


def plot_features_vs_target(X, y, log_features=None, figsize=(12, 6), cmap="viridis", nrows=None, ncols=None):
    log_features = log_features or []

    if not ncols:
        n_cols = len(X.columns)
        ncols = 2

    if not nrows:
        nrows = (n_cols + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, X.columns):
        x = X[col].to_numpy()

        if col in log_features:
            x = np.log(x + 1e-8)  # small offset to avoid log(0)
            x_label = f"log({col})"
        else:
            x_label = col

        hb = ax.hexbin(x, y, gridsize=40, mincnt=1, cmap=cmap)
        ax.set_title(f"{x_label} vs log(y)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("log(y)")
        fig.colorbar(hb, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()