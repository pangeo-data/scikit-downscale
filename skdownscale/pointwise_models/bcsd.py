import collections

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import TimeSynchronousDownscaler
from .utils import QuantileMapper


def MONTH_GROUPER(x):
    return x.month


class BcsdBase(TimeSynchronousDownscaler):
    """ Base class for BCSD model.
    """

    _fit_attributes = ['y_climo_', 'quantile_mappers_']
    _timestep = 'M'

    def __init__(self, time_grouper=MONTH_GROUPER, return_anoms=True, qm_kwargs={}):
        self.time_grouper = time_grouper
        self.return_anoms = return_anoms
        self.qm_kwargs = qm_kwargs

    def _pre_fit(self):
        if isinstance(self.time_grouper, str):
            self.time_grouper_ = pd.Grouper(freq=self.time_grouper)
        else:
            self.time_grouper_ = self.time_grouper

    def _qm_fit_by_group(self, groups):
        """ helper function to fit quantile mappers by group

        Note that we store these mappers for later
        """
        self.quantile_mappers_ = {}
        for key, group in groups:
            self.quantile_mappers_[key] = QuantileMapper(**self.qm_kwargs).fit(group)

    def _qm_transform_by_group(self, groups):
        """ helper function to apply quantile mapping by group

        Note that we recombine the dataframes using pd.concat, there may be a better way to do this
        """

        dfs = []
        for key, group in groups:
            qmapped = self.quantile_mappers_[key].transform(group)
            dfs.append(pd.DataFrame(qmapped, index=group.index, columns=group.columns))
        return pd.concat(dfs).sort_index()


class BcsdPrecipitation(BcsdBase):
    """ Classic BCSD model for Precipitation

    Parameters
    ----------
    time_grouper : str or pd.Grouper, optional
        Pandas time frequency str or Grouper object. Specifies how to group
        time periods. Default is 'M' (e.g. Monthly).
    **qm_kwargs
        Keyword arguments to pass to QuantileMapper.

    Attributes
    ----------
    time_grouper : pd.Grouper
        Linear Regression object.
    quantile_mappers_ : dict
        QuantileMapper objects (one for each time group).
    """

    def fit(self, X, y):
        """ Fit BcsdPrecipitation model

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

        self._pre_fit()
        X, y = self._validate_data(X, y, y_numeric=True)
        if self.n_features_in_ != 1:
            raise ValueError(f'BCSD only supports 1 feature, found {self.n_features_in_}')

        y_groups = y.groupby(self.time_grouper)
        # calculate the climatologies
        self.y_climo_ = y_groups.mean()
        if self.y_climo_.values.min() <= 0:
            raise ValueError('Invalid value in target climatology')

        # fit the quantile mappers
        self._qm_fit_by_group(y_groups)

        return self

    def predict(self, X):
        """Predict using the BcsdPrecipitation model

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Samples.

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        # Bias correction
        # apply quantile mapping by month
        Xqm = self._qm_transform_by_group(X.groupby(self.time_grouper))

        # calculate the anomalies as a ratio of the training data
        if self.return_anoms:
            return self._calc_ratio_anoms(Xqm, self.y_climo_)
        else:
            return Xqm

    def _calc_ratio_anoms(self, obj, climatology):
        dfs = []
        for key, group in obj.groupby(self.time_grouper):
            dfs.append(group / climatology.loc[key].values)

        out = pd.concat(dfs).sort_index()
        assert obj.shape == out.shape
        return out

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_estimators_dtypes': 'BCSD only suppers 1 feature',
                'check_dtype_object': 'BCSD only suppers 1 feature',
                'check_fit_score_takes_y': 'BCSD only suppers 1 feature',
                'check_estimators_fit_returns_self': 'BCSD only suppers 1 feature',
                'check_estimators_fit_returns_self(readonly_memmap=True)': 'BCSD only suppers 1 feature',
                'check_pipeline_consistency': 'BCSD only suppers 1 feature',
                'check_estimators_nan_inf': 'BCSD only suppers 1 feature',
                'check_estimators_overwrite_params': 'BCSD only suppers 1 feature',
                'check_estimators_pickle': 'BCSD only suppers 1 feature',
                'check_fit2d_predict1d': 'BCSD only suppers 1 feature',
                'check_methods_subset_invariance': 'BCSD only suppers 1 feature',
                'check_fit2d_1sample': 'BCSD only suppers 1 feature',
                'check_dict_unchanged': 'BCSD only suppers 1 feature',
                'check_dont_overwrite_parameters': 'BCSD only suppers 1 feature',
                'check_fit_idempotent': 'BCSD only suppers 1 feature',
                'check_n_features_in': 'BCSD only suppers 1 feature',
            },
        }


class BcsdTemperature(BcsdBase):
    def fit(self, X, y):
        """ Fit BcsdTemperature model

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
        self._pre_fit()
        X, y = self._validate_data(X, y, y_numeric=True)
        if self.n_features_in_ != 1:
            raise ValueError(f'BCSD only supports 1 feature, found {self.n_features_in_}')

        # calculate the climatologies
        self._x_climo = X.groupby(self.time_grouper).mean()
        y_groups = y.groupby(self.time_grouper)
        self.y_climo_ = y_groups.mean()

        # fit the quantile mappers
        self._qm_fit_by_group(y_groups)

        return self

    def predict(self, X):
        """ Predict using the BcsdTemperature model

        Parameters
        ----------
        X : DataFrame, shape (n_samples, 1)
            Samples.

        Returns
        -------
        C : pd.DataFrame, shape (n_samples, 1)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = self._check_array(X)

        # X = ensure_samples_features(X)  # don't need????

        # Calculate the 9-year running mean for each month
        def rolling_func(x):
            return x.rolling(9, center=True, min_periods=1).mean()

        X_rolling_mean = X.groupby(self.time_grouper, group_keys=False).apply(rolling_func)

        # calc shift
        # why isn't this working??
        # X_shift = X_rolling_mean.groupby(self.time_grouper) - self._x_climo
        X_shift = self._remove_climatology(X_rolling_mean, self._x_climo)

        # remove shift
        X_no_shift = X - X_shift

        # Bias correction
        # apply quantile mapping by month
        Xqm = self._qm_transform_by_group(X_no_shift.groupby(self.time_grouper))

        # restore the shift
        X_qm_with_shift = X_shift + Xqm
        # calculate the anomalies
        if self.return_anoms:
            return self._remove_climatology(X_qm_with_shift, self.y_climo_)
        else:
            return X_qm_with_shift

    def _remove_climatology(self, obj, climatology):
        dfs = []
        for key, group in obj.groupby(self.time_grouper):
            dfs.append(group - climatology.loc[key].values)

        out = pd.concat(dfs).sort_index()
        assert obj.shape == out.shape
        return out

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_estimators_dtypes': 'BCSD only suppers 1 feature',
                'check_fit_score_takes_y': 'BCSD only suppers 1 feature',
                'check_estimators_fit_returns_self': 'BCSD only suppers 1 feature',
                'check_estimators_fit_returns_self(readonly_memmap=True)': 'BCSD only suppers 1 feature',
                'check_dtype_object': 'BCSD only suppers 1 feature',
                'check_pipeline_consistency': 'BCSD only suppers 1 feature',
                'check_estimators_nan_inf': 'BCSD only suppers 1 feature',
                'check_estimators_overwrite_params': 'BCSD only suppers 1 feature',
                'check_estimators_pickle': 'BCSD only suppers 1 feature',
                'check_fit2d_predict1d': 'BCSD only suppers 1 feature',
                'check_methods_subset_invariance': 'BCSD only suppers 1 feature',
                'check_fit2d_1sample': 'BCSD only suppers 1 feature',
                'check_dict_unchanged': 'BCSD only suppers 1 feature',
                'check_dont_overwrite_parameters': 'BCSD only suppers 1 feature',
                'check_fit_idempotent': 'BCSD only suppers 1 feature',
                'check_n_features_in': 'BCSD only suppers 1 feature',
            },
        }
