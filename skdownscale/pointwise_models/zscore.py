import numpy as np
import pandas as pd
import xarray as xr
from sklearn.utils.validation import check_is_fitted

from .base import TimeSynchronousDownscaler


class ZScoreRegressor(TimeSynchronousDownscaler):
    """ Z Score Regressor bias correction model wrapper

    Apply a scikit-learn model (e.g. Pipeline) point-by-point. The pipeline
    must implement the fit and predict methods.

    Parameters
    ----------
    window_width :  int
        The size of the moving window for statistical analysis. Default is 31
        days.
    """

    _fit_attributes = ['shift_', 'scale_']
    _timestep = 'M'

    def __init__(self, window_width=31):

        assert window_width > 0, window_width
        self.window_width = window_width

    def fit(self, X, y):
        """ Fit Z-Score Model finds the shift and scale parameters
        to inform bias correction.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training historical model data.
        y : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Target measured values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(X, y, y_numeric=True)
        if self.n_features_in_ != 1:
            raise ValueError(f'Zscore only supports 1 feature, found {self.n_features_in_}')

        assert isinstance(X.squeeze(), pd.Series)
        assert isinstance(y.squeeze(), pd.Series)

        X_mean, X_std = _calc_stats(X.squeeze(), self.window_width)
        y_mean, y_std = _calc_stats(y.squeeze(), self.window_width)
        self.fit_stats_dict_ = {
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
        }

        shift, scale = _get_params(X_mean, X_std, y_mean, y_std)

        self.shift_ = shift
        self.scale_ = scale
        return self

    def predict(self, X):
        """ Predict performs the z-score bias correction
        on the future model dataset, X.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training future model data.

        Returns
        -------
        fut_corrected : pd.DataFrame, shape (n_samples, 1)
            Returns corrected values.
        """

        check_is_fitted(self)
        X = self._validate_data(X)

        assert isinstance(X, pd.DataFrame)
        assert X.shape[1] == 1

        name = list(X.keys())[0]

        fut_mean, fut_std, fut_zscore = _get_fut_stats(X.squeeze(), self.window_width)
        shift_expanded, scale_expanded = _expand_params(X.squeeze(), self.shift_, self.scale_)

        fut_mean_corrected, fut_std_corrected = _correct_fut_stats(
            fut_mean, fut_std, shift_expanded, scale_expanded
        )

        self.predict_stats_dict_ = {
            'meani': fut_mean,
            'stdi': fut_std,
            'meanf': fut_mean_corrected,
            'stdf': fut_std_corrected,
        }

        fut_corrected = (fut_zscore * fut_std_corrected) + fut_mean_corrected

        return fut_corrected.to_frame(name)

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_estimators_dtypes': 'Zscore only suppers 1 feature',
                'check_fit_score_takes_y': 'Zscore only suppers 1 feature',
                'check_estimators_fit_returns_self': 'Zscore only suppers 1 feature',
                'check_estimators_fit_returns_self(readonly_memmap=True)': 'Zscore only suppers 1 feature',
                'check_dtype_object': 'Zscore only suppers 1 feature',
                'check_pipeline_consistency': 'Zscore only suppers 1 feature',
                'check_estimators_nan_inf': 'Zscore only suppers 1 feature',
                'check_estimators_overwrite_params': 'Zscore only suppers 1 feature',
                'check_estimators_pickle': 'Zscore only suppers 1 feature',
                'check_fit2d_predict1d': 'Zscore only suppers 1 feature',
                'check_methods_subset_invariance': 'Zscore only suppers 1 feature',
                'check_fit2d_1sample': 'Zscore only suppers 1 feature',
                'check_dict_unchanged': 'Zscore only suppers 1 feature',
                'check_dont_overwrite_parameters': 'Zscore only suppers 1 feature',
                'check_fit_idempotent': 'Zscore only suppers 1 feature',
                'check_n_features_in': 'Zscore only suppers 1 feature',
                'check_fit_check_is_fitted': 'Zscore only suppers 1 feature',
                'check_methods_sample_order_invariance': 'temporal order matters',
            },
        }


def _reshape(da, window_width):
    """
    Helper function for `fit` that splits the year and day
    dimensions of the time-coordinate and bookends the years
    e.g. (Dec15:31 + whole year + Jan1:15) if window_width is 31 days.

    Parameters
    ----------
    da : xr.DataArray, shape (n_samples, )
        Samples
    window_width : int
        The size of the rolling window.

    Returns
    -------
    ds_rsh : xr.Dataset, shape(day: 364 + n_bookend_days, year: n_years)
        Reshaped xr.Dataset
    """

    assert da.ndim == 1

    if 'time' not in da.coords and 'index' in da.coords:
        da = da.rename({'index': 'time'})
    assert 'time' in da.coords

    def split(g):
        return g.rename({'time': 'day'}).assign_coords(day=g.time.dt.dayofyear.values)

    da_split = da.groupby('time.year').map(split)

    early_jans = da_split.isel(day=slice(None, window_width // 2))
    late_decs = da_split.isel(day=slice(-window_width // 2, None))

    da_rsh = xr.concat([late_decs, da_split, early_jans], dim='day')
    return da_rsh


def _calc_stats(series, window_width):
    """
    Helper function for `fit` that calculates the rolling mean and
    standard deviation for each day of the year across all years.

    Parameters
    ----------
    series : pd.Series, shape (n_samples, )
        Samples.
    window_width : int
        The size of the rolling window.

    Returns
    -------
    mean : pd.Series, shape (364, )
        Means for each day of year across all years
    std:  pd.Series, shape (364, )
        Standard deviations for each day of year across all years
    """

    da = series.to_xarray()
    da_rsh = _reshape(da, window_width)

    ds_rolled = da_rsh.rolling(day=window_width, center=True).construct('win_day')

    n = window_width // 2 + 1
    ds_mean = ds_rolled.mean(dim=['year', 'win_day']).isel(day=slice(n, -n))
    ds_std = ds_rolled.std(dim=['year', 'win_day']).isel(day=slice(n, -n))

    mean = ds_mean.to_series()
    std = ds_std.to_series()
    return mean, std


def _get_params(hist_mean, hist_std, meas_mean, meas_std):
    """
    Helper function for `fit` that calculates the shift and scale parameters
    for z-score correction by comparing the historical and measured
    daily means and standard deviations.

    Parameters
    ----------
    hist_mean : pd.Series, shape (364, )
        Mean calculated using the moving window for each day on an average
        year from the historical model.
    hist_std : pd.Series, shape (364, )
        Standard deviation calculated using the moving window for each day on
        an average year from the historical model.
    meas_mean : pd.Series, shape (364, )
        Mean calculated using the moving window for each day on an average year
        from the measurements.
    meas_std : pd.Series, shape (364, )
        Standard deviation calculated using the moving window for each day on
        an average year from the measurements.

    Returns
    -------
    shift : pd.Series, shape (364, )
        The value by which to adjust the future mean.
    scale : pd.Series, shape (364, )
        The value by which to adjust the future standard deviation.
    """

    # TODO: Update docstring to relax the assumption that the year is 364 days long
    # assert len(hist_mean) == 364, len(hist_mean)
    # assert len(hist_std) == 364, len(hist_std)
    # assert len(meas_mean) == 364, len(meas_mean)
    # assert len(meas_std) == 364, len(meas_std)
    assert all([s.ndim for s in [hist_mean, hist_std, meas_mean, meas_std]])

    shift = meas_mean - hist_mean
    scale = meas_std / hist_std
    return shift, scale


# Helpers for Predict
def _get_fut_stats(series, window_width):
    """
    Helper function for `predict` that calculates statistics
    for the future dataset

    Parameters
    ----------
    series : pd.Series, shape (n_samples, )
        Samples
    window_width: int
        The size of the rolling window.

    Returns
    -------
    fut_mean : pd.Series, shape (n_samples, )
        Mean calculated using the moving window for each day of the future
        model.
    fut_std : pd.Series, shape (n_samples, )
        Standard deviation calculated using the moving window for each day of
        the future model.
    fut_zscore: pd.Series, shape (n_samples, )
        Z-Score coefficient calculated by comparing the series values, the
        means, and standared deviations.
    """
    fut_mean = series.rolling(window_width, center=True).mean()
    fut_std = series.rolling(window_width, center=True).std()
    fut_zscore = (series - fut_mean) / fut_std
    return fut_mean, fut_std, fut_zscore


def _expand_params(series, shift, scale):
    """
    Helper function for `predict` that expands the shift and scale parameters
    from a 365-day average year, to the length of the future series.

    Parameters
    ----------
    series : pd.Series, shape (n_samples, )
        Samples.
    shift : pd.Series, shape (364, )
        The value by which to adjust the future mean.
    scale : pd.Series, shape (364, )
        The value by which to adjust the future standard deviation.

    Returns
    -------
    shift_expanded : pd.Series, shape (n_samples, )
        The value by which to adjust the future mean, repeated over the length
        of the series.
    scale_expanded : pd.Series, shape (n_samples, )
        The value by which to adjust the future standard deviation, repeated
        over the length of the series.
    """

    n_samples = len(series)
    len_avgyr = 364 if n_samples > 364 else n_samples
    # TODO: update doc string
    # assert len(shift) == len_avgyr, len(shift)
    # assert len(scale) == len_avgyr, len(scale)

    repeats = int(n_samples / len_avgyr)
    remainder = n_samples % len_avgyr

    inds = np.concatenate([np.tile(np.arange(len_avgyr), repeats), np.arange(remainder)])
    assert len(inds) == n_samples, (len(inds), n_samples)

    shift_expanded = shift.iloc[inds]
    shift_expanded.index = series.index

    scale_expanded = scale.iloc[inds]
    scale_expanded.index = series.index
    return shift_expanded, scale_expanded


def _correct_fut_stats(fut_mean, fut_std, shift_expanded, scale_expanded):
    """
    Helper function for `predict` that adjusts future statistics by shift and
    scale parameters.

    Parameters
    ----------
    fut_mean : pd.Series, shape (n_samples, )
        Mean calculated using the moving window for each day of the future
        model.
    fut_std : pd.Series, shape (n_samples, )
        Standard deviation calculated using the moving window for each day
        of the future model.
    shift_expanded : pd.Series, shape (n_samples, )
        The value by which to adjust the future mean, repeated over the
        length of the Series.
    scale_expanded : pd.Series, shape (n_samples, )
        The value by which to adjust the future standard deviation, repeated over the
        length of the Series.

    Returns
    -------
    fut_mean_corrected : pd.Series, shape (n_samples, )
        Corrected mean for each day of the future model.
    fut_std_corrected : pd.Series, shape (n_samples, )
        Corrected standard deviation for each day of the future model.
    """
    fut_mean_corrected = fut_mean + shift_expanded
    fut_std_corrected = fut_std * scale_expanded
    return fut_mean_corrected, fut_std_corrected
