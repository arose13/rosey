import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


# noinspection PyPep8Naming
class GetSubsetTransform(BaseEstimator, TransformerMixin):
    """
    Selects the columns of a numpy array.
    This allows subsets of X in a pipeline
    """
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X)[:, self.indices]


# noinspection PyPep8Naming
class BinRareTransformer(BaseEstimator, TransformerMixin):
    """
    Makes a new series that is bins with this logic.

    If threshold = 8

    a: 30 -> a: 30
    b: 24 -> b: 24
    c: 50 -> c: 50
    d: 7  -> other: 10
    e: 3  ->
    """
    def __init__(self, threshold: int=100, binned_name='OTHER'):
        self.threshold = threshold
        self.binned_name = binned_name
        self.lookup_ = None  # Dictionary of dictionaries

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.lookup_ = {}

        if isinstance(X, pd.DataFrame):
            for cat_col in X.columns:
                counts = X[cat_col].value_counts()
                categories_all = counts.index.tolist()
                categories_bin = counts[counts <= self.threshold].index.tolist()

                # Create lookup table O(k)
                self.lookup_[cat_col] = dict(zip(categories_all, categories_all))
                for key in categories_bin:
                    self.lookup_[cat_col][key] = self.binned_name

        elif isinstance(X, np.ndarray):
            for cat_col in range(X.shape[1]):
                categories_all, categories_count = np.unique(X, return_counts=True)
                categories_bin = [val for val, n in zip(categories_all, categories_count) if n <= self.threshold]

                # Create lookup table
                self.lookup_[cat_col] = dict(zip(categories_all, categories_all))
                for key in categories_bin:
                    self.lookup_[cat_col][key] = self.binned_name

        else:
            raise TypeError('X must be either a numpy.array or pandas.DataFrame')

        return self

    def transform(self, X):
        # Check for errors
        if self.lookup_ is None:
            raise NotFittedError('You must call .fit() before calling .transform()')

        # Run Transformer
        if isinstance(X, pd.DataFrame):
            for cat_col in X.columns:
                X.loc[:, cat_col] = X[cat_col].copy().map(self.lookup_[cat_col], na_action='ignore')

        elif isinstance(X, np.ndarray):
            for cat_col in range(X.shape[1]):
                binned = np.array(
                    [self.lookup_[cat_col].get(x, default=np.nan if np.isnan(x) else self.binned_name)
                     for x in X[:, cat_col]]
                )
                X[: cat_col] = binned

        else:
            raise TypeError('X must be either a numpy.array or pandas.DataFrame')

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
