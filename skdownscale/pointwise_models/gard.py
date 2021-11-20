import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import default_none_kwargs


def select_analogs(analogs, inds):
    # todo: this is possible with fancy indexing
    out = np.empty(len(analogs))
    for i, ind in enumerate(inds):
        out[i] = analogs[i, ind]
    return out


class AnalogBase(RegressorMixin, BaseEstimator):
    _fit_attributes = ['kdtree_', 'X_', 'y_', 'k_']

    def fit(self, X, y):
        """ Fit Analog model using a KDTree

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, n_features)
            Training data
        y : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(X, y=y, y_numeric=True)
        self.stats_ = {}  # populated in predict methods

        if len(X) >= self.n_analogs:
            self.k_ = self.n_analogs
        else:
            warnings.warn('length of X is less than n_analogs, setting n_analogs = len(X)')
            self.k_ = len(X)

        kdtree_kwargs = default_none_kwargs(self.kdtree_kwargs)
        self.kdtree_ = KDTree(X, **kdtree_kwargs)

        self.X_ = X
        self.y_ = y

        return self


class AnalogRegression(AnalogBase):
    """ AnalogRegression

    Parameters
    ----------
    n_analogs: int
        Number of analogs to use when building linear regression
    thresh: float or int
        Threshold value. If provided, the model will predict:
        1) the probability of this threshold being exceeded, and
        2) the value given the threshold is exceeded
    kdtree_kwargs : dict
        Keyword arguments to pass to the sklearn.neighbors.KDTree constructor
    query_kwargs : dict
        Keyword arguments to pass to the sklearn.neighbors.KDTree.query method
    lr_kwargs : dict
        Keyword arguments to pass to the sklear.linear_model.LinearRegression
        constructor

    Attributes
    ----------
    kdtree_ : sklearn.neighbors.KDTree
        KDTree object
    """

    def __init__(
        self,
        n_analogs=200,
        thresh=None,
        kdtree_kwargs=None,
        query_kwargs=None,
        logistic_kwargs=None,
        lr_kwargs=None,
    ):

        self.n_analogs = n_analogs
        self.thresh = thresh
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs
        self.logistic_kwargs = logistic_kwargs
        self.lr_kwargs = lr_kwargs

    def predict(self, X, return_errors=False, return_exceedance_prob=False):
        """ Predict using the AnalogRegression model

        Parameters
        ----------
        X : DataFrame, shape (n_samples, 1)
            Samples.
        return_errors : bool
        return_exceedance_prob : bool

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        """
        # cannot return error and exceedance prob at the same time
        assert not (return_errors is True and return_exceedance_prob is True)

        # validate input data
        check_is_fitted(self)
        X = check_array(X)

        out = np.empty(len(X))

        # not used if self.thresh = None, instantiating to keep the code clean
        logistic_kwargs = default_none_kwargs(self.logistic_kwargs)
        logistic_model = LogisticRegression(**logistic_kwargs)

        lr_kwargs = default_none_kwargs(self.lr_kwargs)
        lr_model = LinearRegression(**lr_kwargs)

        # TODO - extract from lr_model's below.
        self.stats_['error'] = np.empty(len(X))
        self.stats_['prob'] = np.empty(len(X))

        for i in range(len(X)):
            # predict for this time step
            out[i] = self._predict_one_step(
                logistic_model,
                lr_model,
                X[None, i],
                i,
                return_errors=return_errors,
                return_exceedance_prob=return_exceedance_prob,
            )

        return out

    def _predict_one_step(
        self, logistic_model, lr_model, X, i, return_errors=False, return_exceedance_prob=False
    ):
        if return_exceedance_prob:
            if self.thresh is None:
                return np.nan
            if self.stats_.get('prob', None) is not None:
                return self.stats_['prob'][i]

        if return_errors:
            if self.stats_.get('error', None) is not None:
                return self.stats_['error'][i]

        # get analogs
        query_kwargs = default_none_kwargs(self.query_kwargs)
        inds = self.kdtree_.query(X, k=self.k_, return_distance=False, **query_kwargs).squeeze()

        # extract data to train linear regression model
        x = np.asarray(self.kdtree_.data)[inds]
        y = self.y_[inds]

        # figure out if there's a threshold of interest
        if self.thresh is not None:
            exceed_ind = y > self.thresh
        else:
            exceed_ind = np.array([True] * len(y))

        # train logistic regression model
        if self.thresh is not None:
            binary_y = exceed_ind.astype(int)
            logistic_model.fit(x, binary_y)
            exceedance_prob = logistic_model.predict_proba(X)[:, 0]
            self.stats_['prob'][i] = exceedance_prob[0]

        # train linear regression model otherwise
        lr_model.fit(x[exceed_ind], y[exceed_ind])

        # calculate the rmse of prediction
        y_hat = lr_model.predict(x[exceed_ind])
        error = mean_squared_error(y[exceed_ind], y_hat, squared=False)
        self.stats_['error'][i] = error

        predicted = lr_model.predict(X)

        if return_exceedance_prob:
            return exceedance_prob
        if return_errors:
            return error

        return predicted


class PureAnalog(AnalogBase):
    """ PureAnalog

    Attributes
    ----------
    kdtree_ : sklearn.neighbors.KDTree
        KDTree object
    n_analogs : int
        Number of analogs to use
    thresh : float
        Subset analogs based on threshold
    stats : bool
        Calculate fit statistics during predict step
    kdtree_kwargs : dict
        Dictionary of keyword arguments to pass to cKDTree constructor
    query_kwargs : dict
        Dictionary of keyword arguments to pass to `cKDTree.query`
    """

    def __init__(
        self, n_analogs=200, kind='best_analog', thresh=None, kdtree_kwargs=None, query_kwargs=None,
    ):
        self.n_analogs = n_analogs
        self.kind = kind
        self.thresh = thresh
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs

    def predict(self, X, return_errors=False, return_exceedance_prob=False):
        """Predict using the PureAnalog model

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Samples.

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        """
        # cannot return error and exceedance prob at the same time
        assert not (return_errors is True and return_exceedance_prob is True)

        if return_exceedance_prob:
            if self.thresh is None:
                return np.nan
            elif self.stats_.get('prob', None) is not None:
                return self.stats_['prob']

        elif return_errors:
            if self.stats_.get('error', None) is not None:
                return self.stats_['error']

        # validate input data
        check_is_fitted(self)
        X = check_array(X)

        if self.kind == 'best_analog' or self.n_analogs == 1:
            k = 1
            kind = 'best_analog'
        else:
            k = self.k_
            kind = self.kind

        query_kwargs = default_none_kwargs(self.query_kwargs)
        dist, inds = self.kdtree_.query(X, k=k, **query_kwargs)

        analogs = np.take(self.y_, inds, axis=0)

        if self.thresh is not None:
            # TODO: rethink how the analog threshold is applied.
            # There are certainly edge cases not dealt with properly here
            # particularly in the weight analogs case
            analog_mask = analogs > self.thresh
            masked_analogs = np.where(analog_mask, analogs, np.nan)

        if kind == 'best_analog':
            predicted = analogs[:, 0]

        elif kind == 'sample_analogs':
            # get 1 random index to sample from the analogs
            rand_inds = np.random.randint(low=0, high=k, size=len(X))
            # select the analog now
            predicted = select_analogs(analogs, rand_inds)

        elif kind == 'weight_analogs':
            # take weighted average
            # work around for zero distances (perfect matches)
            tiny = 1e-20
            weights = 1.0 / np.where(dist == 0, tiny, dist)
            if self.thresh is not None:
                predicted = np.average(masked_analogs, weights=weights, axis=1)
            else:
                predicted = np.average(analogs.squeeze(), weights=weights, axis=1)

        elif kind == 'mean_analogs':
            if self.thresh is not None:
                predicted = masked_analogs.mean(axis=1)
            else:
                predicted = analogs.mean(axis=1)

        else:
            raise ValueError('got unexpected kind %s' % kind)

        if self.thresh is not None:
            # for mean/weight cases, this fills nans when all analogs
            # were below thresh
            predicted = np.nan_to_num(predicted, nan=0.0)
            self.stats_['error'] = masked_analogs.std(axis=1)
            # calculate the probability of precip
            self.stats_['prob'] = np.where(analog_mask, 1, 0).mean(axis=1)
        else:
            self.stats_['error'] = analogs.std(axis=1)

        if return_exceedance_prob:
            return self.stats_['prob']
        elif return_errors:
            return self.stats_['error']

        return predicted


class PureRegression(RegressorMixin, BaseEstimator):
    _fit_attributes = ['stats_', 'logistic_model_', 'linear_model_']

    def __init__(
        self, thresh=None, logistic_kwargs=None, linear_kwargs=None,
    ):
        self.thresh = thresh
        self.logistic_kwargs = logistic_kwargs
        self.linear_kwargs = linear_kwargs

    def fit(self, X, y):
        X, y = self._validate_data(X, y=y, y_numeric=True)
        self.stats_ = {}

        if self.thresh is not None:
            exceed_ind = y > self.thresh
            binary_y = exceed_ind.astype(int)
            logistic_kwargs = default_none_kwargs(self.logistic_kwargs)
            self.logistic_model_ = LogisticRegression(**logistic_kwargs).fit(X, binary_y)
        else:
            exceed_ind = [True] * len(y)

        linear_kwargs = default_none_kwargs(self.linear_kwargs)
        self.linear_model_ = LinearRegression(**linear_kwargs).fit(X[exceed_ind], y[exceed_ind])

        y_hat = self.linear_model_.predict(X[exceed_ind])
        self.stats_['error'] = mean_squared_error(y[exceed_ind], y_hat, squared=False)

        return self

    def predict(self, X, return_errors=False, return_exceedance_prob=False):
        check_is_fitted(self)

        # cannot return error and exceedance prob at the same time
        assert not (return_errors is True and return_exceedance_prob is True)

        if return_exceedance_prob:
            if self.thresh is not None:
                return self.logistic_model_.predict_proba(X)[:, 0]
            else:
                return np.nan

        if return_errors:
            return self.stats_['error']

        return self.linear_model_.predict(X)
