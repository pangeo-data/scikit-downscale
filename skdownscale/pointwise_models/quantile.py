import collections
import copy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .trend import LinearTrendTransformer
from .utils import check_max_features, default_none_kwargs

SYNTHETIC_MIN = -1e20
SYNTHETIC_MAX = 1e20

Cdf = collections.namedtuple('CDF', ['pp', 'vals'])


def plotting_positions(n, alpha=0.4, beta=0.4):
    '''Returns a monotonic array of plotting positions.

    Parameters
    ----------
    n : int
        Length of plotting positions to return.
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : ndarray
        Quantile mapped data with shape from `input_data` and probability
        distribution from `data_to_match`.

    See Also
    --------
    scipy.stats.mstats.plotting_positions
    '''
    return (np.arange(1, n + 1) - alpha) / (n + 1.0 - alpha - beta)


class QuantileMapper(TransformerMixin, BaseEstimator):
    """Transform features using quantile mapping.

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

    _fit_attributes = ['x_cdf_fit_']

    def __init__(self, detrend=False, lt_kwargs=None, qt_kwargs=None):

        self.detrend = detrend
        self.lt_kwargs = lt_kwargs
        self.qt_kwargs = qt_kwargs

    def fit(self, X, y=None):
        """Fit the quantile mapping model.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data
        """
        # TO-DO: fix validate data fctn
        X = self._validate_data(X)

        qt_kws = default_none_kwargs(self.qt_kwargs, copy=True)

        if 'n_quantiles' not in qt_kws:
            qt_kws['n_quantiles'] = len(X)

        # maybe detrend the input datasets
        if self.detrend:
            lt_kwargs = default_none_kwargs(self.lt_kwargs)
            x_to_cdf = LinearTrendTransformer(**lt_kwargs).fit_transform(X)
        else:
            x_to_cdf = X

        # calculate the cdfs for X
        # TODO: replace this transformer with something that uses robust
        # empirical cdf plotting positions
        qt = QuantileTransformer(**qt_kws)

        self.x_cdf_fit_ = qt.fit(x_to_cdf)

        return self

    def transform(self, X):
        """Perform the quantile mapping.

        Parameters
        ----------
        X : array_like, shape [n_samples, n_features]
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples, )
            Transformed data
        """
        # validate input data
        check_is_fitted(self)
        # TO-DO: fix validate_data fctn
        X = self._validate_data(X)

        # maybe detrend the datasets
        if self.detrend:
            lt_kwargs = default_none_kwargs(self.lt_kwargs)
            x_trend = LinearTrendTransformer(**lt_kwargs).fit(X)
            x_to_cdf = x_trend.transform(X)
        else:
            x_to_cdf = X

        # do the final mapping
        qt_kws = default_none_kwargs(self.qt_kwargs, copy=True)
        if 'n_quantiles' not in qt_kws:
            qt_kws['n_quantiles'] = len(X)

        x_quantiles = quantile_transform(x_to_cdf, copy=True, **qt_kws)
        x_qmapped = self.x_cdf_fit_.inverse_transform(x_quantiles)

        # add the trend back
        if self.detrend:
            x_qmapped = x_trend.inverse_transform(x_qmapped)

        return x_qmapped

    def _more_tags(self):
        return {'_xfail_checks': {'check_methods_subset_invariance': 'because'}}


class QuantileMappingReressor(RegressorMixin, BaseEstimator):
    """Transform features using quantile mapping.

    Parameters
    ----------
    extrapolate : str, optional
        How to extend the cdfs at the tails. Valid options include {`'min'`, `'max'`, `'both'`, `'1to1'`, `None`}
    n_endpoints : int
        Number of endpoints to include when extrapolating the tails of the cdf

    Attributes
    ----------
    _X_cdf : Cdf
        NamedTuple representing the fit's X cdf
    _y_cdf : Cdf
        NamedTuple representing the fit's y cdf
    """

    _fit_attributes = ['_X_cdf', '_y_cdf']

    def __init__(self, extrapolate=None, n_endpoints=10):
        self.extrapolate = extrapolate
        self.n_endpoints = n_endpoints

        if self.n_endpoints < 2:
            raise ValueError('Invalid number of n_endpoints, must be >= 2')

    def fit(self, X, y, **kwargs):
        """Fit the quantile mapping regression model.

        Parameters
        ----------
        X : array-like, shape  [n_samples, 1]
            Training data.

        Returns
        -------
        self : object
        """
        X = check_array(
            X, dtype='numeric', ensure_min_samples=2 * self.n_endpoints + 1, ensure_2d=True
        )
        y = check_array(
            y, dtype='numeric', ensure_min_samples=2 * self.n_endpoints + 1, ensure_2d=False
        )

        X = check_max_features(X, n=1)

        self._X_cdf = self._calc_extrapolated_cdf(X, sort=True, extrapolate=self.extrapolate)
        self._y_cdf = self._calc_extrapolated_cdf(y, sort=True, extrapolate=self.extrapolate)

        return self

    def predict(self, X, **kwargs):
        """Predict regression for target X.

        Parameters
        ----------
        X : array_like, shape [n_samples, 1]
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples, )
            Predicted data.
        """
        check_is_fitted(self, self._fit_attributes)
        X = check_array(X, ensure_2d=True)

        X = X[:, 0]

        sort_inds = np.argsort(X)

        X_cdf = self._calc_extrapolated_cdf(X[sort_inds], sort=False, extrapolate=self.extrapolate)

        # Fill value for when x < xp[0] or x > xp[-1] (i.e. when X_cdf vals are out of range for self._X_cdf vals)
        left = -np.inf if self.extrapolate in ['min', 'both'] else None
        right = np.inf if self.extrapolate in ['max', 'both'] else None
        # For all values in future X, find the corresponding percentile in historical X
        X_cdf.pp[:] = np.interp(
            X_cdf.vals, self._X_cdf.vals, self._X_cdf.pp, left=left, right=right
        )

        # Extrapolate the tails beyond 1.0 to handle "new extremes", only triggered when the new extremes are even more drastic then
        # the linear extrapolation result from historical X at SYNTHETIC_MIN and SYNTHETIC_MAX
        if np.isinf(X_cdf.pp).any():
            lower_inds = np.nonzero(-np.inf == X_cdf.pp)[0]
            upper_inds = np.nonzero(np.inf == X_cdf.pp)[0]
            model = LinearRegression()
            if len(lower_inds):
                s = slice(lower_inds[-1] + 1, lower_inds[-1] + 1 + self.n_endpoints)
                model.fit(X_cdf.pp[s].reshape(-1, 1), X_cdf.vals[s].reshape(-1, 1))
                X_cdf.pp[lower_inds] = model.predict(X_cdf.vals[lower_inds].reshape(-1, 1))
            if len(upper_inds):
                s = slice(upper_inds[0] - self.n_endpoints, upper_inds[0])
                model.fit(X_cdf.pp[s].reshape(-1, 1), X_cdf.vals[s].reshape(-1, 1))
                X_cdf.pp[upper_inds] = model.predict(X_cdf.vals[upper_inds].reshape(-1, 1))

        # do the full quantile mapping
        y_hat = np.full_like(X, np.nan)
        y_hat[sort_inds] = np.interp(X_cdf.pp, self._y_cdf.pp, self._y_cdf.vals)[1:-1]

        # If extrapolate is 1to1, apply the offset between X and y to the
        # tails of y_hat
        if self.extrapolate == '1to1':
            y_hat = self._extrapolate_1to1(X, y_hat)

        return y_hat

    def _extrapolate_1to1(self, X, y_hat):
        X_fit_len = len(self._X_cdf.vals)
        X_fit_min = self._X_cdf.vals[0]
        X_fit_max = self._X_cdf.vals[-1]

        y_fit_len = len(self._y_cdf.vals)
        y_fit_min = self._y_cdf.vals[0]
        y_fit_max = self._y_cdf.vals[-1]

        # adjust values over fit max
        inds = X > X_fit_max
        if inds.any():
            if X_fit_len == y_fit_len:
                y_hat[inds] = y_fit_max + (X[inds] - X_fit_max)
            elif X_fit_len > y_fit_len:
                X_fit_at_y_fit_max = np.interp(self._y_cdf.pp[-1], self._X_cdf.pp, self._X_cdf.vals)
                y_hat[inds] = y_fit_max + (X[inds] - X_fit_at_y_fit_max)
            elif X_fit_len < y_fit_len:
                y_fit_at_X_fit_max = np.interp(self._X_cdf.pp[-1], self._y_cdf.pp, self._y_cdf.vals)
                y_hat[inds] = y_fit_at_X_fit_max + (X[inds] - X_fit_max)

        # adjust values under fit min
        inds = X < X_fit_min
        if inds.any():
            if X_fit_len == y_fit_len:
                y_hat[inds] = y_fit_min + (X[inds] - X_fit_min)
            elif X_fit_len > y_fit_len:
                X_fit_at_y_fit_min = np.interp(self._y_cdf.pp[0], self._X_cdf.pp, self._X_cdf.vals)
                y_hat[inds] = X_fit_min + (X[inds] - X_fit_at_y_fit_min)
            elif X_fit_len < y_fit_len:
                y_fit_at_X_fit_min = np.interp(self._X_cdf.pp[0], self._y_cdf.pp, self._y_cdf.vals)
                y_hat[inds] = y_fit_at_X_fit_min + (X[inds] - X_fit_min)

        return y_hat

    def _calc_extrapolated_cdf(
        self, data, sort=True, extrapolate=None, pp_min=SYNTHETIC_MIN, pp_max=SYNTHETIC_MAX
    ):
        """ Calculate a new extrapolated cdf

        The goal of this function is to create a CDF with bounds outside the [0, 1] range.
        This allows for quantile mapping beyond observed data points.

        Parameters
        ----------
        data : array_like, shape [n_samples, 1]
            Input data (can be unsorted)
        sort : bool
            If true, sort the data before building the CDF
        extrapolate : str or None
            How to extend the cdfs at the tails. Valid options include {`'min'`, `'max'`, `'both'`, `'1to1'`, `None`}
        pp_min, pp_max : float
            Plotting position min/max values.

        Returns
        -------
        cdf : Cdf (NamedTuple)
        """

        n = len(data)

        # plotting positions
        pp = np.empty(n + 2)
        pp[1:-1] = plotting_positions(n)

        # extended data values (sorted)
        if data.ndim == 2:
            data = data[:, 0]
        if sort:
            data = np.sort(data)
        vals = np.full(n + 2, np.nan)
        vals[1:-1] = data
        vals[0] = data[0]
        vals[-1] = data[-1]

        # Add endpoints to the vector of plotting positions
        if extrapolate in [None, '1to1']:
            pp[0] = pp[1]
            pp[-1] = pp[-2]
        elif extrapolate == 'both':
            pp[0] = pp_min
            pp[-1] = pp_max
        elif extrapolate == 'max':
            pp[0] = pp[1]
            pp[-1] = pp_max
        elif extrapolate == 'min':
            pp[0] = pp_min
            pp[-1] = pp[-2]
        else:
            raise ValueError('unknown value for extrapolate: %s' % extrapolate)

        if extrapolate in ['min', 'max', 'both']:

            model = LinearRegression()

            # extrapolate lower end point
            if extrapolate in ['min', 'both']:
                s = slice(1, self.n_endpoints + 1)
                # fit linear model to first n_endpoints
                model.fit(pp[s].reshape(-1, 1), vals[s].reshape(-1, 1))
                # calculate the data value pp[0]
                vals[0] = model.predict(pp[0].reshape(-1, 1))

            # extrapolate upper end point
            if extrapolate in ['max', 'both']:
                s = slice(-self.n_endpoints - 1, -1)
                # fit linear model to last n_endpoints
                model.fit(pp[s].reshape(-1, 1), vals[s].reshape(-1, 1))
                # calculate the data value pp[-1]
                vals[-1] = model.predict(pp[-1].reshape(-1, 1))

        return Cdf(pp, vals)

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_estimators_dtypes': 'QuantileMappingReressor only suppers 1 feature',
                'check_fit_score_takes_y': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_fit_returns_self': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_fit_returns_self(readonly_memmap=True)': 'QuantileMappingReressor only suppers 1 feature',
                'check_dtype_object': 'QuantileMappingReressor only suppers 1 feature',
                'check_pipeline_consistency': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_nan_inf': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_overwrite_params': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_pickle': 'QuantileMappingReressor only suppers 1 feature',
                'check_fit2d_predict1d': 'QuantileMappingReressor only suppers 1 feature',
                'check_methods_subset_invariance': 'QuantileMappingReressor only suppers 1 feature',
                'check_fit2d_1sample': 'QuantileMappingReressor only suppers 1 feature',
                'check_dict_unchanged': 'QuantileMappingReressor only suppers 1 feature',
                'check_dont_overwrite_parameters': 'QuantileMappingReressor only suppers 1 feature',
                'check_fit_idempotent': 'QuantileMappingReressor only suppers 1 feature',
                'check_n_features_in': 'QuantileMappingReressor only suppers 1 feature',
                'check_estimators_empty_data_messages': 'skip due to odd sklearn string matching in unit test',
                'check_regressors_train': 'QuantileMappingReressor only suppers 1 feature',
                'check_regressors_train(readonly_memmap=True)': 'QuantileMappingReressor only suppers 1 feature',
                'check_regressors_train(readonly_memmap=True,X_dtype=float32)': 'QuantileMappingReressor only suppers 1 feature',
                'check_regressor_data_not_an_array': 'QuantileMappingReressor only suppers 1 feature',
                'check_regressors_no_decision_function': 'QuantileMappingReressor only suppers 1 feature',
                'check_supervised_y_2d': 'QuantileMappingReressor only suppers 1 feature',
                'check_regressors_int': 'QuantileMappingReressor only suppers 1 feature',
                'check_methods_sample_order_invariance': 'QuantileMappingReressor only suppers 1 feature',
                'check_fit_check_is_fitted': 'QuantileMappingReressor only suppers 1 feature',
            },
        }


class EquidistantCdfMatcher(QuantileMappingReressor):
    """Transform features using equidistant CDF matching, a version of quantile mapping that preserves the difference or ratio between X_test and X_train.

    Parameters
    ----------
    extrapolate : str, optional
        How to extend the cdfs at the tails. Valid options include {`'min'`, `'max'`, `'both'`, `'1to1'`, `None`}
    n_endpoints : int
        Number of endpoints to include when extrapolating the tails of the cdf

    Attributes
    ----------
    _X_cdf : Cdf
        NamedTuple representing the fit's X cdf
    _y_cdf : Cdf
        NamedTuple representing the fit's y cdf
    """

    _fit_attributes = ['_X_cdf', '_y_cdf']

    def __init__(self, kind='difference', extrapolate=None, n_endpoints=10, max_ratio=None):
        if kind not in ['difference', 'ratio']:
            raise NotImplementedError('kind must be either difference or ratio')
        self.kind = kind
        self.extrapolate = extrapolate
        self.n_endpoints = n_endpoints
        # MACA seems to have a max ratio for precip at 5.0
        self.max_ratio = max_ratio

        if self.n_endpoints < 2:
            raise ValueError('Invalid number of n_endpoints, must be >= 2')

    def predict(self, X, **kwargs):
        """Predict regression for target X.

        Parameters
        ----------
        X : array_like, shape [n_samples, 1]
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples, )
            Predicted data.
        """
        check_is_fitted(self, self._fit_attributes)
        X = check_array(X, ensure_2d=True)

        X = X[:, 0]

        sort_inds = np.argsort(X)

        X_cdf = self._calc_extrapolated_cdf(X[sort_inds], sort=False, extrapolate=self.extrapolate)
        X_train_vals = np.interp(x=X_cdf.pp, xp=self._X_cdf.pp, fp=self._X_cdf.vals)

        # generate y value as historical y plus/multiply by quantile difference
        if self.kind == 'difference':
            diff = X_cdf.vals - X_train_vals
            sorted_y_hat = np.interp(x=X_cdf.pp, xp=self._y_cdf.pp, fp=self._y_cdf.vals) + diff
        elif self.kind == 'ratio':
            ratio = X_cdf.vals / X_train_vals
            if self.max_ratio is not None:
                ratio = np.min(ratio, self.max_ratio)
            sorted_y_hat = np.interp(x=X_cdf.pp, xp=self._y_cdf.pp, fp=self._y_cdf.vals) * ratio

        # put things into the right order
        y_hat = np.full_like(X, np.nan)
        y_hat[sort_inds] = sorted_y_hat[1:-1]

        # If extrapolate is 1to1, apply the offset between X and y to the
        # tails of y_hat
        if self.extrapolate == '1to1':
            y_hat = self._extrapolate_1to1(X, y_hat)

        return y_hat


class TrendAwareQuantileMappingRegressor(RegressorMixin, BaseEstimator):
    """Experimental meta estimator for performing trend-aware quantile mapping

    Parameters
    ----------
    qm_estimator : object, default=None
        Regressor object such as ``QuantileMappingReressor``.
    """

    def __init__(self, qm_estimator=None, trend_transformer=None):
        self.qm_estimator = qm_estimator
        if trend_transformer is None:
            self.trend_transformer = LinearTrendTransformer()

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like, shape  [n_samples, n_features]
            Training data.

        Returns
        -------
        self : object
        """
        self._X_mean_fit = X.mean()
        self._y_mean_fit = y.mean()

        y_trend = copy.deepcopy(self.trend_transformer)
        y_detrend = y_trend.fit_transform(y)

        X_trend = copy.deepcopy(self.trend_transformer)
        x_detrend = X_trend.fit_transform(X)

        self.qm_estimator.fit(x_detrend, y_detrend)

        return self

    def predict(self, X):
        """Predict regression for target X.

        Parameters
        ----------
        X : array_like, shape [n_samples, n_features]
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples, )
            Predicted data.
        """
        X_trend = copy.deepcopy(self.trend_transformer)
        x_detrend = X_trend.fit_transform(X)

        y_hat = self.qm_estimator.predict(x_detrend).reshape(-1, 1)

        # add the mean and trend back
        # delta: X (predict) - X (fit) + y --> projected change + historical obs mean
        delta = (X.mean().values - self._X_mean_fit.values) + self._y_mean_fit.values

        # calculate the trendline
        # TODO: think about how this would need to change if we're using a rolling average trend
        trendline = X_trend.trendline(X)
        trendline -= trendline.mean()  # center at 0

        # apply the trend and delta
        y_hat += trendline + delta

        return y_hat
