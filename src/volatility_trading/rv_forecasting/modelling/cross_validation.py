import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator for time series.

    - Folds are contiguous in time (no shuffle).
    - Around each test fold we:
        * purge `purge_gap` samples immediately before and after the test fold
        * embargo a fraction `embargo` of samples AFTER the test fold

    This is useful when targets or features use lookbacks / lookaheads
    (e.g. rolling RV), so that training data does not leak future info
    from the test window.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    purge_gap : int, default=0
        Number of samples to drop from the training set immediately
        before and after the test block in index space.
    embargo : float, default=0.0
        Fraction (0..1) of the dataset to embargo *after* each test fold.
        Example: embargo=0.01 with 10,000 samples → next 100 samples
        after the test block are removed from the training set.
    """

    def __init__(self, n_splits=5, purge_gap=0, embargo=0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0.")
        if not (0.0 <= embargo < 1.0):
            raise ValueError("embargo must be in [0.0, 1.0).")

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices.

        X is assumed to be time-ordered already: X[0] is earliest,
        X[-1] is latest.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # build contiguous folds (like standard KFold with shuffle=False)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop

        n_embargo = int(np.floor(n_samples * self.embargo))

        for test_idx in folds:
            test_start = test_idx[0]
            test_end = test_idx[-1]

            # start with all indices
            train_idx = indices.copy()

            # remove test indices themselves
            train_idx = np.setdiff1d(train_idx, test_idx, assume_unique=False)

            # purge region before and after test
            if self.purge_gap > 0:
                left = max(test_start - self.purge_gap, 0)
                right = min(test_end + self.purge_gap, n_samples - 1)
                purge_region = np.arange(left, right + 1)
                train_idx = np.setdiff1d(train_idx, purge_region, assume_unique=False)

            # embargo region after test
            if n_embargo > 0:
                emb_start = min(test_end + 1, n_samples)
                emb_end = min(emb_start + n_embargo, n_samples)
                if emb_start < emb_end:
                    embargo_region = np.arange(emb_start, emb_end)
                    train_idx = np.setdiff1d(
                        train_idx, embargo_region, assume_unique=False
                    )

            yield np.sort(train_idx), np.sort(test_idx)
