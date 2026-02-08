import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_features_vs_target(
    X,
    y,
    log_features=None,
    sqrt_features=None,
    figsize=(12, 6),
    cmap="viridis",
    nrows=None,
    ncols=None,
):
    log_features = log_features or []
    sqrt_features = sqrt_features or []

    # layout
    if not ncols:
        n_cols = len(X.columns)
        ncols = 2
    if not nrows:
        nrows = (n_cols + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, X.columns):
        x = X[col].to_numpy()

        # choose transform / label
        if col in log_features:
            x = np.log(x + 1e-8)  # small offset to avoid log(0)
            x_label = f"log({col})"
        elif col in sqrt_features:
            x = np.sqrt(np.clip(x, 0.0, None))
            x_label = f"sqrt({col})"
        else:
            x_label = col

        hb = ax.hexbin(x, y, gridsize=40, mincnt=1, cmap=cmap)
        ax.set_title(f"{x_label} vs log(y)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("log(y)")
        fig.colorbar(hb, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()


def plot_macro_block(X_macro: pd.DataFrame):
    """
    Quick overview of macro predictors:
    - Top: Treasury yields (levels)
    - Bottom: Term spread + credit spreads
    """
    cols_rates = [c for c in ["DGS10", "DGS2", "DGS3MO"] if c in X_macro.columns]
    cols_spreads = [
        c for c in ["term_spread_10y_3m", "HY_OAS", "IG_OAS"] if c in X_macro.columns
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 1) Yields
    if cols_rates:
        X_macro[cols_rates].plot(ax=axes[0], lw=1.3)
        axes[0].set_title("Treasury Yields")
        axes[0].set_ylabel("%")
        axes[0].grid(alpha=0.3, linestyle="--")

    # 2) Spreads
    if cols_spreads:
        X_macro[cols_spreads].plot(ax=axes[1], lw=1.3)
        axes[1].set_title("Term & Credit Spreads")
        axes[1].set_ylabel("% / bps")
        axes[1].grid(alpha=0.3, linestyle="--")

    # x-axis formatting (years)
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(X, bins=40, figsize=(12, 5), nrows=None, ncols=None):
    X_df = pd.DataFrame(X)
    cols = X_df.columns
    n_features = len(cols)

    if ncols is None:
        ncols = min(4, n_features)
    if nrows is None:
        nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        data = X_df[col].dropna().to_numpy()
        ax.hist(data, bins=bins, alpha=0.7)

        sk = skew(data)
        kt = kurtosis(data)

        ax.set_title(col)
        ax.tick_params(axis="both", labelsize=8)

        ax.text(
            0.97,
            0.97,
            f"Skew={sk:.2f}\nKurt={kt:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # hide unused axes if any
    for ax in axes[len(cols) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_hist_transform(
    series, use_log=False, use_sqrt=False, winsorize=None, figsize=(8, 3)
):
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # raw
    axes[0].hist(raw, bins=40, alpha=0.7, color="steelblue")
    axes[0].set_title(f"Distribution of {series.name} (raw)")
    axes[0].text(
        0.95,
        0.95,
        f"Skew={stats_raw[0]:.2f}\nKurt={stats_raw[1]:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # transformed
    axes[1].hist(transformed, bins=40, alpha=0.7, color="darkorange")
    axes[1].set_title(f"Distribution of {series.name} ({tlabel})")
    axes[1].text(
        0.95,
        0.95,
        f"Skew={stats_trans[0]:.2f}\nKurt={stats_trans[1]:.2f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def plot_mean_std_importance(
    df,
    value_col="mean",
    std_col="std",
    feature_col="feature",
    title="",
    top_n=20,
    sort_abs=True,
    abs_values=False,
    figsize=(8, 5),
):
    data = df.copy()

    # sort
    if sort_abs:
        data = data.reindex(data[value_col].abs().sort_values(ascending=False).index)
    else:
        data = data.sort_values(value_col, ascending=False)

    data = data.head(top_n)

    vals = data[value_col].values
    if abs_values:
        vals = np.abs(vals)

    plt.figure(figsize=figsize)
    plt.barh(
        data[feature_col],
        vals,
        xerr=data[std_col].values,
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Mean importance" + (" (|value|)" if abs_values else ""))
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_purged_kfold_splits(cv, X, y):
    import matplotlib.dates as mdates

    dates = X.index

    plt.figure(figsize=(10, 3))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        y_level = fold
        plt.scatter(
            dates[tr],
            np.full_like(tr, y_level),
            marker="s",
            s=4,
            label="train" if fold == 0 else None,
            c="blue",
        )
        plt.scatter(
            dates[val],
            np.full_like(val, y_level),
            marker="s",
            s=10,
            label="val" if fold == 0 else None,
            c="orange",
        )

    plt.yticks(
        range(cv.get_n_splits()), [f"Fold {k}" for k in range(cv.get_n_splits())]
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlabel("Date")
    plt.title("Purged K-Fold scheme (train vs validation over time)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_lasso_coef_paths(
    coefs,
    feature_names,
    top_n=10,
    folds=None,
    figsize=(10, 5),
):
    coefs = np.asarray(coefs)
    feature_names = np.asarray(feature_names)
    n_folds, n_features = coefs.shape

    if folds is None:
        folds = np.arange(n_folds)
    else:
        folds = np.asarray(folds)

    # rank features by mean |coef|
    mean_abs = np.abs(coefs).mean(axis=0)
    idx_top = np.argsort(-mean_abs)[:top_n]

    coefs_top = coefs[:, idx_top]
    feats_top = feature_names[:top_n]

    plt.figure(figsize=figsize)
    for j, feat in enumerate(feats_top):
        plt.plot(folds, coefs_top[:, j], marker="o", label=feat)

    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("CV fold")
    plt.ylabel("Coefficient value")
    plt.title(f"Lasso coefficient paths across folds (top {top_n} by |β|)")
    plt.xticks(folds, [str(k) for k in folds])
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


def plot_subperiod_comparison(perf, subperiods, figsize=(12, 6)):
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
    axes = axes.ravel()

    for ax, (start, end, label) in zip(axes, subperiods):
        mask = (perf.index >= start) & (perf.index <= end)
        df_sub = perf.loc[mask]

        y_true_sub = df_sub["y_true"]
        y_har_vix_sub = df_sub["har_vix"]
        y_iv_sub = df_sub["naive_iv"]

        ax.plot(y_true_sub.index, y_true_sub, label="True log RV", alpha=0.6)
        ax.plot(y_har_vix_sub.index, y_har_vix_sub, label="HAR-RV-VIX", alpha=0.9)
        ax.plot(y_iv_sub.index, y_iv_sub, label="Naive IV", alpha=0.9, linestyle="--")

        ax.set_ylabel("log 21D RV")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)

        # --- monthly ticks, every 3 months ---
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("True vs HAR-RV-VIX vs Naive IV – sub-period comparison", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_model_comparison_ts(
    y, y_pred_1, y_pred_2, label_1, label_2, start=None, end=None
):
    # Restrict to window if start/end are provided
    if start is not None or end is not None:
        y = y.loc[start:end]
        y_pred_1 = y_pred_1.loc[start:end]
        y_pred_2 = y_pred_2.loc[start:end]

    # --- 1) Time-series overlay ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index, y.values, label="True RV")
    ax.plot(y_pred_1.index, y_pred_1.values, label=label_1, alpha=0.8)
    ax.plot(y_pred_2.index, y_pred_2.values, label=label_2, alpha=0.6)
    ax.set_title("True vs predicted 21D RV")
    ax.set_xlabel("Date")
    ax.set_ylabel("RV")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison_scatter(
    y,
    y_pred_1,
    y_pred_2,
    label_1,
    label_2,
):
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF {title}")

    plot_pacf(
        series.dropna(), lags=lags, ax=axes[1], method="ywm"
    )  # Yule-Walker-M estimator
    axes[1].set_title(f"PACF {title}")

    plt.tight_layout()
    plt.show()
