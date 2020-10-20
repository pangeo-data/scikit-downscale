import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from .utils import Cdf, check_max_features, plotting_positions

SYNTHETIC_MIN = -1e20
SYNTHETIC_MAX = 1e20


class QuantileMappingReressor(BaseEstimator, RegressorMixin):

    _fit_attributes = ['_X_cdf', '_y_cdf']

    def __init__(self, extrapolate=None, n_endpoints=10, cdf_kwargs={}):
        self.extrapolate = extrapolate
        self.n_endpoints = n_endpoints
        self.cdf_kwargs = cdf_kwargs

        if self.n_endpoints < 2:
            raise ValueError('Invalid number of n_endpoints, must be >= 2')

    def fit(self, X, y, **kwargs):

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2 * self.n_endpoints + 1)
        X = check_max_features(X)

        self._X_cdf = self._calc_extrapolated_cdf(X, sort=True, extrapolate=self.extrapolate)
        self._y_cdf = self._calc_extrapolated_cdf(y, sort=True, extrapolate=self.extrapolate)

        return self

    def predict(self, X, **kwargs):

        check_is_fitted(self, self._fit_attributes)
        X = check_array(X)

        X = X[:, 0]

        sort_inds = np.argsort(X)

        X_cdf = self._calc_extrapolated_cdf(X[sort_inds], sort=False, extrapolate=self.extrapolate)

        left = -np.inf if self.extrapolate in ['min', 'both'] else None
        right = np.inf if self.extrapolate in ['max', 'both'] else None
        X_cdf.pp[:] = np.interp(
            X_cdf.vals, self._X_cdf.vals, self._X_cdf.pp, left=left, right=right
        )

        # Extrapolate the tails beyond 1.0 to handle "new extremes"
        if np.isinf(X_cdf.pp).any():
            lower_inds = np.nonzero(-np.inf == X_cdf.pp)[0]
            upper_inds = np.nonzero(np.inf == X_cdf.pp)[0]
            model = LinearRegression()
            if len(lower_inds):
                s = slice(lower_inds[-1] + 1, lower_inds[-1] + 1 + self.n_endpoints)
                model.fit(X_cdf.pp[s], X_cdf.vals[s])
                X_cdf.pp[lower_inds] = model.predict(X_cdf.vals[lower_inds])
            if len(upper_inds):
                s = slice(upper_inds[0] - self.n_endpoints, upper_inds[0])
                model.fit(X_cdf.pp[s], X_cdf.vals[s])
                X_cdf.pp[upper_inds] = model.predict(X_cdf.vals[upper_inds])

        # do the full quantile mapping
        y_hat = np.full_like(X, np.nan)
        y_hat[sort_inds] = np.interp(X_cdf.pp, self._y_cdf.pp, self._y_cdf.vals)[1:-1]

        # If extrapolate is 1to1, apply the offset between ref and like to the
        # tails of y_hat
        if self.extrapolate == '1to1':
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
                    X_fit_at_y_fit_max = np.interp(
                        self._y_cdf.pp[-1], self._X_cdf.pp, self._X_cdf.vals
                    )
                    y_hat[inds] = y_fit_max + (X[inds] - X_fit_at_y_fit_max)
                elif X_fit_len < y_fit_len:
                    y_fit_at_X_fit_max = np.interp(
                        self._X_cdf.pp[-1], self._y_cdf.pp, self._y_cdf.vals
                    )
                    y_hat[inds] = y_fit_at_X_fit_max + (X[inds] - X_fit_max)

            # adjust values under fit min
            inds = X < X_fit_min
            if inds.any():
                if X_fit_len == y_fit_len:
                    y_hat[inds] = y_fit_min + (X[inds] - X_fit_min)
                elif X_fit_len > y_fit_len:
                    X_fit_at_y_fit_min = np.interp(
                        self._y_cdf.pp[0], self._X_cdf.pp, self._X_cdf.vals
                    )
                    y_hat[inds] = X_fit_min + (X[inds] - X_fit_at_y_fit_min)
                elif X_fit_len < y_fit_len:
                    y_fit_at_X_fit_min = np.interp(
                        self._X_cdf.pp[0], self._y_cdf.pp, self._y_cdf.vals
                    )
                    y_hat[inds] = y_fit_at_X_fit_min + (X[inds] - X_fit_min)

        return y_hat

    def _calc_extrapolated_cdf(
        self, data, sort=True, extrapolate=None, pp_min=SYNTHETIC_MIN, pp_max=SYNTHETIC_MAX
    ):

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
            },
        }
