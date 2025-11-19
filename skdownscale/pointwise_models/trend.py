from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import default_none_kwargs


class LinearTrendTransformer(TransformerMixin, BaseEstimator):
    """Transform features by removing linear trends.

    Uses Ordinary least squares Linear Regression as implemented in
    sklear.linear_model.LinearRegression.

    Parameters
    ----------
    **lr_kwargs
        Keyword arguments to pass to sklearn.linear_model.LinearRegression

    Attributes
    ----------
    lr_model_ : sklearn.linear_model.LinearRegression
        Linear Regression object.
    """

    def __init__(self, lr_kwargs: dict[str, Any] | None = None) -> None:
        self.lr_kwargs = lr_kwargs

    def _validate_data(
        self, X: ArrayLike, y: ArrayLike | None = None, reset: bool = True, **check_params: Any
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """Validate input data using sklearn's validate_data."""
        return validate_data(self, X=X, y=y, reset=reset, **check_params)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> LinearTrendTransformer:
        """Compute the linear trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.
        """
        X = self._validate_data(X)
        kwargs = default_none_kwargs(self.lr_kwargs)
        self.lr_model_ = LinearRegression(**kwargs)
        self.lr_model_.fit(np.arange(len(X)).reshape(-1, 1), X)
        return self

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Perform transformation by removing the trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be detrended.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        return X - self.trendline(X)

    def inverse_transform(self, X: ArrayLike) -> NDArray[Any]:
        """Add the trend back to the data.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be transformed back.
        """
        # validate input data
        check_is_fitted(self)
        X = self._validate_data(X)
        return X + self.trendline(X)

    def trendline(self, X: ArrayLike) -> NDArray[Any]:
        """helper function to calculate a linear trendline"""
        X = self._validate_data(X)
        return self.lr_model_.predict(np.arange(len(X)).reshape(-1, 1))

    def __sklearn_tags__(self):
        from dataclasses import replace

        tags = super().__sklearn_tags__()
        # Mark as skipping certain tests due to temporal sensitivity
        tags = replace(tags, _skip_test='Temporal transformer - sample order matters')
        return tags
