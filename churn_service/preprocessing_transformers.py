"""Scikit-learn compatible transformers used by the churn preprocessor."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to lower/upper percentiles fit on training data."""

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X = check_array(X, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = check_array(X, ensure_all_finite="allow-nan", copy=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {X.shape[1]}"
            )
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array(
            [f"x{i}" for i in range(self.n_features_in_)], dtype=object
        )
