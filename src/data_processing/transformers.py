import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
