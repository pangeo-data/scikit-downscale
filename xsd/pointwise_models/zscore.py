import pandas as pd
import numpy as np
import xarray as xr

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel


class ZScoreRegressor(LinearModel, RegressorMixin):
    """ Z Score Regressor bias correction model wrapper

    Apply a scikit-learn model (e.g. Pipeline) point-by-point. The pipeline
    must implement the fit and predict methods.

    Parameters:
    ----------
    window_width :  int
        The size of the moving window for statistical analysis.
        Default is 31 days.
    var_str :  str
        The key associated with the target dataframe variable.
        Default is 'foo'
    """

    def __init__(self, window_width=31, var_str='foo'):

        self.window_width = window_width
        self.var_str = var_str

    def fit(self, X, y):
        """ Fit Z-Score Model finds the shift and scale parameters
        to inform bias correction.

        Parameters:
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training historical model data
        y : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Target measured values.

        Returns:
        -------
        self : returns an instance of self.
        """

        X_mean, X_std = _calc_stats(X, self.window_width)
        y_mean, y_std = _calc_stats(y, self.window_width)

        shift, scale = _get_params(X_mean, X_std, y_mean, y_std)

        self.shift = shift
        self.scale = scale

        return self

    def predict(self, X):
        """ Predict performs the z-score bias correction
        on the future model dataset, X.

        Parameters:
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training historical model data

        Returns:
        -------
        fut_corrected : pd.DataFrame, shape (n_samples, 1)
            Returns corrected values.
        """

        fut_mean, fut_std, fut_zscore = _get_fut_stats(X, self.window_width)
        shift_expanded, scale_expanded = _expand_params(X, self.var_str, self.shift, self.scale)

        fut_mean_corrected, fut_std_corrected = _correct_fut_stats(fut_mean, fut_std, shift_expanded, scale_expanded)
        fut_corrected = (fut_zscore * fut_std_corrected) + fut_mean_corrected
        return fut_corrected


def _reshape(ds, window_width):
    """
    Helper function for `fit` that splits the year and day
    dimensions of the time-coordinate and bookends the years
    e.g. (Dec15:31 + whole year + Jan1:15) if window_width is 31 days.

    Parameters:
    ----------
    ds : xr.Dataset, shape (n_samples, 1)
        Samples
    window_width: int
        the size of the rolling window.

    Returns:
    -------
    ds_rsh : xr.Dataset, shape(day: 364 + n_bookend_days, year: n_years)
        Reshaped xr.Dataset
    """

    if 'time' not in ds.coords and 'index' in ds.coords:
        ds = ds.rename({'index': 'time'})
    assert 'time' in ds.coords

    split = lambda g: (g.rename({'time': 'day'})
                       .assign_coords(day=g.time.dt.dayofyear.values))
    ds_split = ds.groupby('time.year').apply(split)

    early_Jans = ds_split.isel(day=slice(None, window_width // 2))
    late_Decs = ds_split.isel(day=slice(-window_width // 2, None))

    ds_rsh = xr.concat([late_Decs, ds_split, early_Jans], dim='day')
    return ds_rsh


def _calc_stats(df, window_width):
    """
    Helper function for `fit` that calculates the rolling mean and
    standard deviation for each day of the year across all years.

    Parameters:
    ----------
    df : pd.DataFrame, shape (n_samples, 1)
        Samples.
    window_width: int
        the size of the rolling window.

    Returns:
    -------
    df_mean : pd.DataFrame, shape (364, 1)
        Means for each day of year across all years
    df_std:  pd.DataFrame, shape (364, 1)
        Standard deviations for each day of year across all years
    """

    ds = df.to_xarray()
    ds_rsh = _reshape(ds, window_width)

    ds_rolled = ds_rsh.rolling(day=window_width, center=True).construct('win_day')

    n = window_width // 2 + 1
    ds_mean = ds_rolled.mean(dim=['year', 'win_day']).isel(day=slice(n, -n))
    ds_std = ds_rolled.std(dim=['year', 'win_day']).isel(day=slice(n, -n))

    df_mean = ds_mean.to_dataframe()
    df_std = ds_std.to_dataframe()
    return df_mean, df_std


def _get_params(hist_mean, hist_std, meas_mean, meas_std):
    """
    Helper function for `fit` that calculates the shift and scale parameters
    for z-score correction by comparing the historical and measured
    daily means and standard deviations.

    Parameters:
    ----------
    hist_mean : pd.DataFrame, shape (364, 1)
        Mean calculated using the moving window
        for each day on an average year from the historical
        model
    hist_std : pd.DataFrame, shape (364, 1)
        Standard deviation calculated using the
        moving window for each day on an average year
        from the historical model
    meas_mean : pd.DataFrame, shape (364, 1)
        Mean calculated using the moving window
        for each day on an average year from the measurements
    meas_std : pd.DataFrame, shape (364, 1)
        Standard deviation calculated using the
        moving window for each day on an average year
        from the measurements

    Returns:
    -------
    shift : pd.DataFrame, shape (364, 1)
        The value by which to adjust the future mean.
    scale : pd.DataFrame, shape (364, 1)
        The value by which to adjust the future standard deviation.
    """
    shift = meas_mean - hist_mean
    scale = meas_std / hist_std
    return shift, scale


# Helpers for Predict
def _get_fut_stats(df, window_width):
    """
    Helper function for `predict` that calculates statistics
    for the future dataset

    Parameters:
    ----------
    df : pd.Dataframe, shape (n_samples, 1)
        Samples
    window_width: int
        The size of the rolling window.

    Returns:
    -------
    fut_mean : pd.Dataframe, shape (n_samples, 1)
        Mean calculated using the moving window
        for each day of the future model
    fut_std : pd.Dataframe, shape (n_samples, 1)
        Standard deviation calculated using the
        moving window for each day of the future model
    fut_zscore: pd.Dataframe, shape (n_samples, 1)
        Z-Score coefficient calculated by comparing
        the dataframe values, the means, and standared
        deviations.
    """
    fut_mean = df.rolling(window_width, center=True).mean()
    fut_std = df.rolling(window_width, center=True).std()
    fut_zscore = (df - fut_mean) / fut_std
    return fut_mean, fut_std, fut_zscore


def _expand_params(df, var_str, shift, scale):
    """
    Helper function for `predict` that expands the shift and scale parameters
    from a 365-day average year, to the length of the future dataframe.

    Parameters:
    ----------
    df : pd.DataFrame, shape (n_samples, 1)
        Samples.
    var_str :  str
        The key associated with the target dataframe variable.
    shift : pd.DataFrame, shape (364, 1)
        The value by which to adjust the future mean.
    scale : pd.DataFrame, shape (364, 1)
        The value by which to adjust the future standard deviation.

    Returns:
    -------
    shift_expanded : pd.Dataframe, shape (n_samples, 1)
        The value by which to adjust the future mean,
        repeated over the length of the dataframe.
    scale_expanded : pd.Dataframe, shape (n_samples, 1)
        The value by which to adjust the future standard deviation,
        repeated over the length of the dataframe.
    """
    repeats = int(df.shape[0] / shift[var_str].shape[0])
    remainder = df.shape[0] % shift[var_str].shape[0]

    sh_repeated = np.tile(shift[var_str], repeats)
    sc_repeated = np.tile(scale[var_str], repeats)
    sh_remaining = shift[var_str][0:remainder].values
    sc_remaining = scale[var_str][0:remainder].values

    data_shift_expanded = np.concatenate((sh_repeated, sh_remaining))
    data_scale_expanded = np.concatenate((sc_repeated, sc_remaining))

    shift_expanded = xr.DataArray(data_shift_expanded, name=var_str, dims=['index'], coords={'index': df.index}).to_dataframe()
    scale_expanded = xr.DataArray(data_scale_expanded, name=var_str, dims=['index'], coords={'index': df.index}).to_dataframe()

    return shift_expanded, scale_expanded


def _correct_fut_stats(fut_mean, fut_std, shift_expanded, scale_expanded):
    """
    Helper function for `predict` that adjusts future statistics
    by shift and scale parameters.

    Parameters:
    ----------
    fut_mean : pd.Dataframe, shape (n_samples, 1)
        Mean calculated using the moving window
        for each day of the future model
    fut_std : pd.Dataframe, shape (n_samples, 1)
        Standard deviation calculated using the
        moving window for each day of the future model
    shift_expanded : pd.Dataframe, shape (n_samples, 1)
        The value by which to adjust the future mean,
        repeated over the length of the dataframe.
    scale_expanded : pd.Dataframe, shape (n_samples, 1)
        The value by which to adjust the future standard deviation,
        repeated over the length of the dataframe.

    Returns:
    -------
    fut_mean_corrected : pd.Dataframe, shape (n_samples, 1)
        Corrected mean for each day of the future model
    fut_std_corrected : pd.Dataframe, shape (n_samples, 1)
        Corrected standard deviation for each day of the future model
    """
    fut_mean_corrected = fut_mean + shift_expanded
    fut_std_corrected = fut_std * scale_expanded
    return fut_mean_corrected, fut_std_corrected
