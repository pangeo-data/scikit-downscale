import collections

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer, quantile_transform

Cdf = collections.namedtuple("CDF", ["pp", "vals"])


class LinearTrendTransformer(TransformerMixin, BaseEstimator):
    """ Transform features by removing linear trends.

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

    def __init__(self, **lr_kwargs):
        self.lr_kwargs = lr_kwargs

    def fit(self, X):
        """ Compute the linear trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.
        """
        self.lr_model_ = LinearRegression(**self.lr_kwargs)
        self.lr_model_.fit(np.arange(len(X)).reshape(-1, 1), X)
        return self

    def transform(self, X):
        """ Perform transformation by removing the trend.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be detrended.
        """
        return X - self._trendline(X)

    def inverse_transform(self, X):
        """ Add the trend back to the data.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            The data that should be transformed back.
        """
        return X + self._trendline(X)

    def _trendline(self, X):
        """ helper function to calculate a linear trendline """
        return self.lr_model_.predict(np.arange(len(X)).reshape(-1, 1))


class QuantileMapper(BaseEstimator, TransformerMixin):
    """ Transform features using quantile mapping.

    Parameters
    ----------
    detrend : boolean, optional
        If True, detrend the data before quantile mapping and add the trend
        back after transforming. Default is False.
    lt_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the LinearTrendTransformer
    qm_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the QuantileMapper

    Attributes
    ----------
    x_cdf_fit_ : QuantileTransformer
        QuantileTranform for fit(X)
    """

    def __init__(self, detrend=False, lt_kwargs={}, qt_kwargs={}):

        self.lt_kwargs = lt_kwargs
        self.qt_kwargs = qt_kwargs

        self.detrend = detrend

    def fit(self, X):
        """ Fit the quantile mapping model.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.
        """
        X = ensure_samples_features(X)

        qt_kws = self.qt_kwargs.copy()

        if "n_quantiles" not in qt_kws:
            qt_kws["n_quantiles"] = len(X)

        # maybe detrend the input datasets
        if self.detrend:
            x_to_cdf = LinearTrendTransformer(**self.lt_kwargs).fit_transform(X)
        else:
            x_to_cdf = X

        # calculate the cdfs for X
        # TODO: replace this transformer with something that uses robust
        # empirical cdf plotting positions
        self.x_cdf_fit_ = QuantileTransformer(**qt_kws).fit(x_to_cdf)

        return self

    def transform(self, X):
        """ Perform the quantile mapping.

        Parameters
        ----------
        X : array_like, shape [n_samples, n_features]
            Samples.
        """
        X = ensure_samples_features(X)

        # maybe detrend the datasets
        if self.detrend:
            x_trend = LinearTrendTransformer(**self.lt_kwargs).fit(X)
            x_to_cdf = x_trend.transform(X)
        else:
            x_to_cdf = X

        # do the final mapping
        qt_kws = self.qt_kwargs.copy()
        if "n_quantiles" not in qt_kws:
            qt_kws["n_quantiles"] = len(X)

        x_quantiles = quantile_transform(x_to_cdf, copy=True, **qt_kws)
        x_qmapped = self.x_cdf_fit_.inverse_transform(x_quantiles)

        # add the trend back
        if self.detrend:
            x_qmapped = x_trend.inverse_transform(x_qmapped)

        return x_qmapped


def ensure_samples_features(obj):
    """ helper function to ensure sammples conform to sklearn format
    requirements
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return obj
        if obj.ndim == 1:
            return obj.reshape(-1, 1)
    return obj  # hope for the best, probably better to raise an error here
