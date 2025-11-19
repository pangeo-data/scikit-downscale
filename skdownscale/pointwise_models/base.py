from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y


class TimeSynchronousDownscaler(BaseEstimator):
    def _check_X_y(
        self, X: ArrayLike, y: ArrayLike, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            pd.testing.assert_index_equal(X.index, y.index)
            check_X_y(X, y)  # this may be inefficient
        else:
            X, y = check_X_y(X, y)
            warnings.warn('X and y do not have pandas DateTimeIndexes, making one up...')
            index = pd.date_range(periods=len(X), start='1950', freq='MS')
            X = pd.DataFrame(X, index=index)
            y = pd.DataFrame(y, index=index)
        return X, y

    def _check_array(self, array: ArrayLike, **kwargs: Any) -> pd.DataFrame:
        if isinstance(array, pd.DataFrame):
            check_array(array)
        else:
            array = check_array(array)
            warnings.warn('array does not have a pandas DateTimeIndex, making one up...')
            index = pd.date_range(periods=len(array), start='1950', freq=self._timestep)
            array = pd.DataFrame(array, index=index)

        return array

    def _check_n_features(self, X: ArrayLike, reset: bool) -> None:
        """Check and set n_features_in_ attribute.

        Parameters
        ----------
        X : array-like
            Input data
        reset : bool
            Whether to reset n_features_in_ or check consistency
        """
        n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1

        if reset:
            self.n_features_in_ = n_features
        elif hasattr(self, 'n_features_in_'):
            if self.n_features_in_ != n_features:
                raise ValueError(
                    f'X has {n_features} features, but {self.__class__.__name__} '
                    f'was fitted with {self.n_features_in_} features.'
                )

    def __sklearn_tags__(self):
        """Get estimator tags for sklearn 1.6+.

        Returns
        -------
        tags : Tags
            Tags object with estimator metadata.
        """
        from dataclasses import replace

        tags = super().__sklearn_tags__()
        # Update target_tags to indicate y is not required by default
        tags = replace(tags, target_tags=replace(tags.target_tags, required=False))
        return tags

    def _validate_data(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        reset: bool = True,
        validate_separately: bool = False,
        **check_params: Any,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,), default=None
            The targets. If None, `check_array` is called on `X` and
            `check_X_y` is called otherwise.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.
        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if `y` is not None.
        """

        if y is None:
            tags = self.__sklearn_tags__()
            if tags.target_tags.required:
                raise ValueError(
                    f'This {self.__class__.__name__} estimator '
                    f'requires y to be passed, but the target y is None.'
                )
            X = self._check_array(X, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = self._check_array(X, **check_X_params)
                y = self._check_array(y, **check_y_params)
            else:
                X, y = self._check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out
