import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log(np.clip(X, self.eps, None))

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        # keep original names
        return np.asarray(input_features, dtype=str)


class SqrtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sqrt(np.clip(X, 0.0, None))

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        # keep original names
        return np.asarray(input_features, dtype=str)


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q_low_ = np.quantile(X, self.lower, axis=0)
        self.q_high_ = np.quantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.q_low_, self.q_high_)

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(self.n_features_in_)])
        # keep original names
        return np.asarray(input_features, dtype=str)


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 log_features=None,
                 sqrt_features=None,
                 winsor_features=None,
                 winsor_sqrt_features=None,
                 scale=True):
        """
        log_features: list of cols → log only
        sqrt_features: list of cols → sqrt only
        winsor_features: list of ( [low, high], [col1, col2, ...] ) → winsor only
        winsor_sqrt_features: list of ( [low, high], [col1, col2, ...] )
            → winsorize, then sqrt

        Each feature must appear in at most ONE of:
        - log_features
        - sqrt_features
        - any winsor_features group
        - any winsor_sqrt_features group
        Others are passed through unchanged.
        """
        self.log_features = [] if log_features is None else log_features
        self.sqrt_features = [] if sqrt_features is None else sqrt_features
        self.winsor_features = [] if winsor_features is None else winsor_features
        self.winsor_sqrt_features = [] if winsor_sqrt_features is None else winsor_sqrt_features
        self.scale = scale

        self.preprocessor_ = None
        self.pipeline_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        all_cols = list(X_df.columns)

        # intersect lists with actual columns
        log_cols = [c for c in self.log_features if c in all_cols]
        sqrt_cols = [c for c in self.sqrt_features if c in all_cols]

        winsor_features_clean = []
        for (q_range, cols) in self.winsor_features:
            cols_present = [c for c in cols if c in all_cols]
            winsor_features_clean.append((q_range, cols_present))

        winsor_sqrt_features_clean = []
        for (q_range, cols) in self.winsor_sqrt_features:
            cols_present = [c for c in cols if c in all_cols]
            winsor_sqrt_features_clean.append((q_range, cols_present))

        # collect all winsor columns actually present
        winsor_cols_all = []
        for (q_range, cols) in winsor_features_clean:
            winsor_cols_all.extend(cols)

        winsor_sqrt_cols_all = []
        for (q_range, cols) in winsor_sqrt_features_clean:
            winsor_sqrt_cols_all.extend(cols)

        used = (
            set(log_cols)
            | set(sqrt_cols)
            | set(winsor_cols_all)
            | set(winsor_sqrt_cols_all)
        )
        passthrough_cols = [c for c in all_cols if c not in used]

        transformers = []

        if log_cols:
            transformers.append(("log", LogTransformer(), log_cols))

        if sqrt_cols:
            transformers.append(("sqrt", SqrtTransformer(), sqrt_cols))

        for (q_range, cols) in winsor_features_clean:
            if not cols:
                continue
            low, high = q_range
            name = f"winsor_{low}_{high}".replace(".", "p")
            transformers.append((name, Winsorizer(lower=low, upper=high), cols))

        for (q_range, cols) in winsor_sqrt_features_clean:
            if not cols:
                continue
            low, high = q_range
            name = f"winsor_sqrt_{low}_{high}".replace(".", "p")
            ws_pipeline = Pipeline([
                ("winsor", Winsorizer(lower=low, upper=high)),
                ("sqrt", SqrtTransformer()),
            ])
            transformers.append((name, ws_pipeline, cols))

        if passthrough_cols:
            transformers.append(("passthrough", "passthrough", passthrough_cols))

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        steps = [("preprocess", self.preprocessor_)]
        if self.scale:
            steps.append(("scaler", StandardScaler()))

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X_df, y)

        try:
            self.feature_names_out_ = self.preprocessor_.get_feature_names_out()
        except AttributeError:
            self.feature_names_out_ = None

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return self.pipeline_.transform(X_df)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.feature_names_out_