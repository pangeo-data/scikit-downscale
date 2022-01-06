import collections

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .base import TimeSynchronousDownscaler
from .groupers import DAY_GROUPER, MONTH_GROUPER, PaddedDOYGrouper
from .quantile import QuantileMapper
from .utils import default_none_kwargs, ensure_samples_features


class BcsdBase(TimeSynchronousDownscaler):
    """Base class for BCSD model."""

    _fit_attributes = ['y_climo_', 'quantile_mappers_']
    _timestep = 'M'

    def __init__(
        self,
        time_grouper=MONTH_GROUPER,
        climate_trend_grouper=DAY_GROUPER,
        climate_trend=MONTH_GROUPER,
        return_anoms=True,
        qm_kwargs=None,
    ):

        self.time_grouper = time_grouper
        self.climate_trend_grouper = climate_trend_grouper
        self.climate_trend = climate_trend
        self.return_anoms = return_anoms
        self.qm_kwargs = qm_kwargs

    def _pre_fit(self):
        if isinstance(self.time_grouper, str):
            if self.time_grouper == 'daily_nasa-nex':
                self.time_grouper = PaddedDOYGrouper
                self.timestep = 'daily'
            else:
                self.time_grouper_ = pd.Grouper(freq=self.time_grouper)
                self.timestep = 'monthly'
        else:
            self.time_grouper_ = self.time_grouper
            self.timestep = 'monthly'

    def _create_groups(self, df, climate_trend=False):
        """helper function to create groups by either daily or month"""
        if self.timestep == 'monthly':
            return df.groupby(self.time_grouper)
        elif self.timestep == 'daily':
            if climate_trend:
                # group by day only rather than also +/- offset days
                return df.groupby(self.climate_trend_grouper)
            else:
                return self.time_grouper(df)
        else:
            raise TypeError('unexpected time grouper type %s' % self.time_grouper)

    def _qm_fit_by_group(self, groups):
        """helper function to fit quantile mappers by group

        Note that we store these mappers for later
        """
        self.quantile_mappers_ = {}
        qm_kwargs = default_none_kwargs(self.qm_kwargs)
        for key, group in groups:
            self.quantile_mappers_[key] = QuantileMapper(**qm_kwargs).fit(group)

    def _qm_transform_by_group(self, groups):
        """helper function to apply quantile mapping by group

        Note that we recombine the dataframes using pd.concat, there may be a better way to do this
        """

        dfs = []
        for key, group in groups:
            qmapped = self.quantile_mappers_[key].transform(group)
            dfs.append(pd.DataFrame(qmapped, index=group.index, columns=group.columns))
        return pd.concat(dfs).sort_index()

    def _remove_climatology(self, obj, climatology, climate_trend=False):
        """helper function to remove climatologies"""
        dfs = []
        for key, group in self._create_groups(obj, climate_trend):
            if self.timestep == 'monthly':
                dfs.append(group - climatology.loc[key].values)
            elif self.timestep == 'daily':
                dfs.append(group - climatology.loc[key])

        result = pd.concat(dfs).sort_index()
        assert obj.shape == result.shape
        return result


class BcsdPrecipitation(BcsdBase):
    """Classic BCSD model for Precipitation

    Parameters
    ----------
    time_grouper : str or pd.Grouper, optional
        Pandas time frequency str or Grouper object. Specifies how to group
        time periods. Default is 'M' (e.g. Monthly).
    qm_kwargs : dict
        Keyword arguments to pass to QuantileMapper.

    Attributes
    ----------
    time_grouper : pd.Grouper
        Linear Regression object.
    quantile_mappers_ : dict
        QuantileMapper objects (one for each time group).
    """

    def fit(self, X, y):
        """Fit BcsdPrecipitation model

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
        # TO-DO: set n_features_n attribute
        if self.n_features_in_ != 1:
            raise ValueError(f'BCSD only supports 1 feature, found {self.n_features_in_}')

        y_groups = self._create_groups(y)
        # calculate the climatologies
        self.y_climo_ = y_groups.mean()

        if self.return_anoms and self.y_climo_.values.min() <= 0:
            raise ValueError('Invalid value in target climatology')

        # fit the quantile mappers
        # TO-DO: do we need to detrend the data before fitting the quantile mappers??
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
        # apply quantile mapping by month or day
        Xqm = self._qm_transform_by_group(self._create_groups(X, climate_trend=True))

        # calculate the anomalies as a ratio of the training data
        if self.return_anoms:
            return self._calc_ratio_anoms(Xqm, self.y_climo_)
        else:
            return Xqm

    def _calc_ratio_anoms(self, obj, climatology, climate_trend=False):
        """helper function for dividing day groups by climatology"""
        dfs = []
        for key, group in self._create_groups(obj, climate_trend):
            if self.timestep == 'monthly':
                dfs.append(group / climatology.loc[key].values)
            else:
                dfs.append(group / climatology.loc[key])

        result = pd.concat(dfs).sort_index()
        assert obj.shape == result.shape

        return result

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
                'check_fit_check_is_fitted': 'BCSD only suppers 1 feature',
                'check_methods_sample_order_invariance': 'temporal order matters',
            },
        }


class BcsdTemperature(BcsdBase):
    def fit(self, X, y):
        """Fit BcsdTemperature model

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
        # TO-DO: set n_features_in attribute
        if self.n_features_in_ != 1:
            raise ValueError(f'BCSD only supports up to 4 features, found {self.n_features_in_}')

        # make groups for day or month
        y_groups = self._create_groups(y)

        # calculate the climatologies
        self._x_climo = self._create_groups(X).mean()
        self.y_climo_ = y_groups.mean()

        # fit the quantile mappers
        self._qm_fit_by_group(y_groups)

        return self

    def predict(self, X):
        """Predict using the BcsdTemperature model

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

        # Calculate the 9-year running mean for each month
        def rolling_func(x):
            return x.rolling(9, center=True, min_periods=1).mean()

        X_rolling_mean = X.groupby(self.climate_trend, group_keys=False).apply(rolling_func)

        # remove climatology from 9-year monthly mean climate trend
        X_shift = self._remove_climatology(X_rolling_mean, self._x_climo, climate_trend=True)

        # remove shift from model data
        X_no_shift = X - X_shift

        # Bias correction
        # apply quantile mapping by month or day
        Xqm = self._qm_transform_by_group(self._create_groups(X_no_shift, climate_trend=True))

        # restore the climate trend
        X_qm_with_shift = X_shift + Xqm

        # return bias corrected absolute values or calculate the anomalies
        if self.return_anoms:
            return self._remove_climatology(X_qm_with_shift, self.y_climo_)
        else:
            return X_qm_with_shift

    def _remove_climatology(self, obj, climatology, climate_trend=False):
        """helper function to remove climatologies"""
        dfs = []
        for key, group in self._create_groups(obj, climate_trend):
            if self.timestep == 'monthly':
                dfs.append(group - climatology.loc[key].values)
            elif self.timestep == 'daily':
                dfs.append(group - climatology.loc[key].values)

        result = pd.concat(dfs).sort_index()
        if obj.shape != result.shape:
            raise ValueError('shape of climo is not equal to input array')
        return result

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
                'check_fit_check_is_fitted': 'BCSD only suppers 1 feature',
                'check_methods_sample_order_invariance': 'temporal order matters',
            },
        }
