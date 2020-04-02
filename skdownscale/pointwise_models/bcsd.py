import collections

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils.validation import check_is_fitted

from .utils import QuantileMapper, ensure_samples_features
from .groupers import PaddedDOYGrouper


def MONTH_GROUPER(x):
    return x.month


class BcsdBase(LinearModel, RegressorMixin):
    """ Base class for BCSD model.
    """

    _fit_attributes = ["y_climo_", "quantile_mappers_"]

    def __init__(self, time_grouper=MONTH_GROUPER, **qm_kwargs):
        if isinstance(time_grouper, str):
            if time_grouper == 'daily_nasa-nex':
                self.time_grouper = PaddedDOYGrouper
                self.timestep = 'daily'
                self.upsample = True
            else:
                self.time_grouper = pd.Grouper(freq=time_grouper)
        else:
            self.time_grouper = time_grouper
            self.timestep = 'monthly'
            self.upsample = False

        self.qm_kwargs = qm_kwargs
        
    def _create_groups(self, df):
        """ helper function to create groups by either daily or month,
        depending on whether we are bias correcting daily or monthly data
        """
        if self.timestep == 'monthly':
            return df.groupby(self.time_grouper)
        elif self.timestep== 'daily':
            return self.time_grouper(df)
        else:
            raise TypeError('unexpected time grouper type %s' % self.time_grouper)
        
    def _create_temperature_climatology_groups(self, df):
        """ helper function to create climatology groups for either daily or monthly data. 
        Note: the reason we need this function in addition to the above one is that for BC'ing 
        daily data, we still want to calculate the 9-year running average of monthly data, so we 
        can't use the above function. Instead we want to groupby month if we are BCing monthly data, 
        and we want to resample to monthly data if we are BCing daily data. 
        
        Note: this is variable specific, since I think we want to sum for precip rather than 
        taking the mean. 
        """
        if isinstance(self.time_grouper, pd.Grouper):
            upsample = False
            return (df.groupby(self.time_grouper), upsample)
        elif isinstance(self.time_grouper, XsdGroupGeneratorBase):
            df_monthly = df.resample('M').mean()
            upsample = True
            return (df_monthly(pd.Grouper(freq='M')), upsample)
        else:
            raise TypeError('unexpected time grouper type %s' % self.time_grouper)

    def _qm_fit_by_group(self, groups):
        """ helper function to fit quantile mappers by group

        Note that we store these mappers for later
        """
        self.quantile_mappers_ = {}
        for key, group in groups:
            data = ensure_samples_features(group)
            self.quantile_mappers_[key] = QuantileMapper(**self.qm_kwargs).fit(data)

    def _qm_transform_by_group(self, groups):
        """ helper function to apply quantile mapping by group

        Note that we recombine the dataframes using pd.concat, there may be a better way to do this
        """

        dfs = []
        for key, group in groups:
            data = ensure_samples_features(group)
            qmapped = self.quantile_mappers_[key].transform(data)
            dfs.append(pd.DataFrame(qmapped, index=group.index, columns=data.columns))
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
        y_groups = y.groupby(self.time_grouper)
        # calculate the climatologies
        self.y_climo_ = y_groups.mean()
        if self.y_climo_.values.min() <= 0:
            raise ValueError("Invalid value in target climatology")

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
        check_is_fitted(self, self._fit_attributes)
        X = ensure_samples_features(X)

        # Bias correction
        # apply quantile mapping by month
        Xqm = self._qm_transform_by_group(X.groupby(self.time_grouper))

        # calculate the anomalies as a ratio of the training data
        return self._calc_ratio_anoms(Xqm, self.y_climo_)

    def _calc_ratio_anoms(self, obj, climatology):
        dfs = []
        for key, group in obj.groupby(self.time_grouper):
            dfs.append(group / climatology.loc[key].values)

        out = pd.concat(dfs).sort_index()
        assert obj.shape == out.shape
        return out


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
        # calculate the climatologies
        # NOTE: here we want the means of daily or monthly groups. In the `predict` function when 
        # we extract large scale GCM trends, then we want the monthly means regardless of whether we 
        # are bias correcting daily or monthly data (for the NASA-NEX implementation of BCSD at least)
        self._x_climo = self._create_groups(X).mean()
        
        # make groups for day or month 
        y_groups = self._create_groups(y)
        
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
        check_is_fitted(self, self._fit_attributes)
        X = ensure_samples_features(X)

        # Calculate the 9-year running mean for each month
        def rolling_func(x):
            return x.rolling(9, center=True, min_periods=1).mean()
        
        # X_climo_groups, upsample = self._create_temperature_climatology_groups(X)
        # X_rolling_mean = X.groupby(self.time_grouper).apply(rolling_func)
        X_rolling_mean = X.groupby(MONTH_GROUPER).apply(rolling_func)
        # X_rolling_mean = X_climo_groups.apply(rolling_func)
        
        # if X is daily data, need to upsample X_rolling_mean to daily 
        if self.upsample:
            X_rolling_mean = X_rolling_mean.resample('D').mean()

        # calc shift
        # why isn't this working??
        # X_shift = X_rolling_mean.groupby(self.time_grouper) - self._x_climo
        X_shift = self._remove_climatology(X_rolling_mean, self._x_climo)

        # remove shift
        #X_no_shift = X - X_shift
        X_no_shift = check_datetime_index(X, self.timestep) - check_datetime_index(X_shift, self.timestep)

        # Bias correction
        # apply quantile mapping by month
        # Xqm = self._qm_transform_by_group(X_no_shift.groupby(self.time_grouper))
        Xqm = self._qm_transform_by_group(self._create_groups(X_no_shift))

        # restore the shift
        X_qm_with_shift = X_shift + Xqm
        # calculate the anomalies
        # return self._remove_climatology(X_qm_with_shift, self.y_climo_)
        return X_rolling_mean, self._x_climo, self.y_climo_, Xqm, X_no_shift, X, X_shift, self._remove_climatology(X_qm_with_shift, self.y_climo_)

    def _remove_climatology(self, obj, climatology):
        dfs = []
        #for key, group in obj.groupby(self.time_grouper):
        for key, group in self._create_groups(obj):
            if self.timestep == 'monthly':
                dfs.append(group - climatology.loc[key].values)
            elif self.timestep == 'daily':
                dfs.append(group - climatology.loc[key])

        out = pd.concat(dfs).sort_index()
        assert obj.shape == out.shape
        return out
