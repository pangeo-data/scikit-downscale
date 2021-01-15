import numpy as np
import pandas as pd

from .utils import default_none_kwargs


class GroupedRegressor:
    """ Grouped Regressor

    Wrapper supporting fitting seperate estimators distinct groups

    Parameters
    ----------
    estimator : object
        Estimator object such as derived from `BaseEstimator`. This estimator will be fit to each group
    fit_grouper : object
        Grouper object, such as `pd.Grouper` or `PaddedDOYGrouper` used to split data into groups during fitting.
    predict_grouper : object, func, str
        Grouper object, such as `pd.Grouper` used to split data into groups during prediction.
    estimator_kwargs : dict
        Keyword arguments to pass onto the `estimator`'s contructor.
    fit_grouper_kwargs : dict
        Keyword arguments to pass onto the `fit_grouper`s contructor.
    predict_grouper_kwargs : dict
        Keyword arguments to pass onto the `predict_grouper`s contructor.
    """

    def __init__(
        self,
        estimator,
        fit_grouper,
        predict_grouper,
        estimator_kwargs=None,
        fit_grouper_kwargs=None,
        predict_grouper_kwargs=None,
    ):

        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs

        self.fit_grouper = fit_grouper
        self.fit_grouper_kwargs = fit_grouper_kwargs

        self.predict_grouper = predict_grouper
        self.predict_grouper_kwargs = predict_grouper_kwargs

    def fit(self, X, y, **fit_kwargs):
        """ Fit the grouped regressor

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Training data
        y : pd.Series or pd.DataFrame, shape (n_samples, ) or (n_samples, n_targets)
            Target values
        **fit_kwargs
            Additional keyword arguments to pass onto the estimator's fit method

        Returns
        -------
        self : returns an instance of self.
        """
        fit_grouper_kwargs = default_none_kwargs(self.fit_grouper_kwargs)
        x_groups = self.fit_grouper(X.index, **fit_grouper_kwargs).groups
        y_groups = self.fit_grouper(y.index, **fit_grouper_kwargs).groups

        self.targets_ = list(y.keys())
        estimator_kwargs = default_none_kwargs(self.estimator_kwargs)
        self.estimators_ = {key: self.estimator(**estimator_kwargs) for key in x_groups}

        for x_key, x_inds in x_groups.items():
            y_inds = y_groups[x_key]

            self.estimators_[x_key].fit(X.iloc[x_inds], y.iloc[y_inds], **fit_kwargs)

        return self

    def predict(self, X):
        """ Predict estimator target for X

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Training data

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.

        """
        predict_grouper_kwargs = default_none_kwargs(self.predict_grouper_kwargs)
        grouper = X.groupby(self.predict_grouper, **predict_grouper_kwargs)

        result = np.empty((len(X), len(self.targets_)))
        for key, inds in grouper.indices.items():
            result[inds, ...] = self.estimators_[key].predict(X.iloc[inds])

        return result


class PaddedDOYGrouper:
    """ Grouper to group an Index by day-of-year +/ pad

    Parameters
    ----------
    index : pd.DatetimeIndex
        Pandas DatetimeIndex to be grouped.
    window : int
        Size of the padded offset for each day of year.
    """

    def __init__(self, index, window):
        self.index = index
        self.window = window

        idoy = index.dayofyear
        n = idoy.max()

        # day-of-year x day-of-year groups
        temp_groups = np.zeros((n, n), dtype=np.bool)
        for i in range(n):
            inds = np.arange(i - self.window, i + self.window + 1)
            inds[inds < 0] += n
            inds[inds > n - 1] -= n
            temp_groups[i, inds] = True

        arr = temp_groups[idoy - 1]
        self._groups = {doy: np.nonzero(arr[:, doy - 1])[0] for doy in range(1, n + 1)}

    @property
    def groups(self):
        """ Dict {doy -> group indicies}."""
        return self._groups
