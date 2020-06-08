import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
from sklearn.utils.validation import check_is_fitted

from .base import AbstractDownscaler
from .utils import ensure_samples_features


class AnalogBase(AbstractDownscaler):
    _fit_attributes = ['kdtree_', 'y_']

    def fit(self, X, y):
        """ Fit Analog model using a KDTree

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training data
        y : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        if len(X) < self.n_analogs:
            warnings.warn('length of X is less than n_analogs, setting n_analogs = len(X)')
            self.n_analogs = len(X)

        self.kdtree_ = KDTree(X, **self.kdtree_kwargs)
        self.y_ = y

        return self


class AnalogRegression(AnalogBase):
    """ AnalogRegression

    Parameters
    ----------
    n_analogs: int
        Number of analogs to use when building linear regression
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

    def __init__(self, n_analogs=200, kdtree_kwargs={}, query_kwargs={}, lr_kwargs={}):

        self.n_analogs = n_analogs
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs
        self.lr_kwargs = lr_kwargs
        self.lr_model = LinearRegression(**self.lr_kwargs)

    def predict(self, X):
        """ Predict using the AnalogRegression model

        Parameters
        ----------
        X : DataFrame, shape (n_samples, 1)
            Samples.

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        """
        check_is_fitted(self, self._fit_attributes)

        predicted = np.empty(len(X))

        # TODO - extract from lr_model's below.
        self.stats = {}

        for i, (_, row) in enumerate(X.iterrows()):
            # predict for this time step
            predicted[i] = self._predict_one_step(ensure_samples_features(row.values))

        return predicted

    def _predict_one_step(self, X):
        # get analogs
        inds = self.kdtree_.query(
            X, k=self.n_analogs, return_distance=False, **self.query_kwargs
        ).squeeze()

        # extract data to train linear regression model
        x = np.asarray(self.kdtree_.data)[inds]
        y = self.y_.values[inds]

        # train linear regression model
        self.lr_model.fit(x, y)

        # predict for this time step
        predicted = self.lr_model.predict(ensure_samples_features(X))
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
        self,
        n_analogs=200,
        kind='best_analog',
        thresh=None,
        stats=True,
        kdtree_kwargs={},
        query_kwargs={},
    ):
        self.thresh = thresh
        self.stats = stats
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs

        if kind == 'best_analog' or n_analogs == 1:
            self.n_analogs = 1
            self.kind = 'best_analog'
        else:
            self.n_analogs = n_analogs
            self.kind = kind

    def predict(self, X):
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
        check_is_fitted(self, self._fit_attributes)
        self.stats_ = {}

        dist, inds = self.kdtree_.query(X, k=self.n_analogs, **self.query_kwargs)
        dist = dist.squeeze()
        inds = inds.squeeze()

        analogs = np.take(self.y_.values, inds, axis=0)

        if self.thresh is not None:
            # TODO: rethink how the analog threshold is applied.
            # There are certainly edge cases not dealt with properly here
            # particularly in the weight analogs case
            analog_mask = analogs > self.thresh
            masked_analogs = analogs[analog_mask]

        if self.kind == 'best_analog':
            predicted = analogs

        elif self.kind == 'sample_analogs':
            # get 1 random index to sample from the analogs
            rand_inds = np.random.randint(low=0, high=self.n_analogs, size=len(X))
            # select the analog now
            predicted = select_analogs(analogs, rand_inds)

        elif self.kind == 'weight_analogs':
            # take weighted average
            # work around for zero distances (perfect matches)
            tiny = 1e-20
            weights = 1.0 / np.where(dist == 0, tiny, dist)
            if self.thresh:
                predicted = np.average(masked_analogs, weights=weights, axis=1)
            else:
                predicted = np.average(analogs.squeeze(), weights=weights, axis=1)

        elif self.kind == 'mean_analogs':
            if self.thresh is not None:
                predicted = masked_analogs.mean(axis=1)
            else:
                predicted = analogs.mean(axis=1)

        else:
            raise ValueError('got unexpected kind %s' % self.kind)

        if self.thresh is not None:
            # for mean/weight cases, this fills nans when all analogs
            # were below thresh
            predicted = np.nan_to_num(predicted, nan=0.0)

        if self.stats:
            # calculate the standard deviation of the anlogs
            if self.thresh is None:
                self.stats_['error'] = analogs.std(axis=1)
            else:
                self.stats_['error'] = analogs.where(analog_mask).std(axis=1)
                # calculate the probability of precip
                self.stats_['pop'] = np.where(analog_mask, 1, 0).mean(axis=1)

        return predicted


def select_analogs(analogs, inds):
    # todo: this is possible with fancy indexing
    out = np.empty(len(analogs))
    for i, ind in enumerate(inds):
        out[i] = analogs[i, ind]
    return out
