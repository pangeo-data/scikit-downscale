import pandas as pd
import numpy as np
import xarray as xr

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel


class ZScoreRegressor(LinearModel, RegressorMixin):
    """ Z Score Regressor bias correction model wrapper

    Apply a scikit-learn model (e.g. Pipeline) point-by-point. The pipeline
    must implement the fit and predict methods.
    """

    def __init__(self, window_width=31, var_str='U'):

        self.window_width = window_width
        self.var_str = var_str


    def fit(self, X, y):
        """ Fit Z-Score Model finds the shift and scale parameters
        to inform bias correction.
             
        Parameters
        ----------
        X : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Training historical model data
        y : pd.Series or pd.DataFrame, shape (n_samples, 1)
            Target measured values.
        window_width :  The size of the moving window for 
            statistical analysis. Default is 31 days.

        Returns
        -------
        self : returns an instance of self.
        """

        X_mean, X_std = _calc_stats(X, self.window_width)
        y_mean, y_std  = _calc_stats(y, self.window_width)
        
        shift, scale = _get_params(X_mean, X_std, y_mean, y_std)

        self.shift = shift
        self.scale = scale
        
        return self    
        
        
    def predict(self, X):
        """ Predict performs the z-score bias correction 
        on the future model dataset, X.
        
        Parameters
        ----------
        X : DataFrame, shape (n_samples, 1)
            Samples.
        window_width : The size of the moving window for 
            statistical analysis. Default is 31 days.
        var_str :  The key associated with the target 
            dataframe variable

        Returns
        -------
        fut_corrected : pd.DataFrame, shape (n_samples, 1)
            Returns corrected values.
        """

         if 'U' not in X.columns.values:
            self.var_str =X.columns.values[0]

        fut_mean, fut_std, fut_zscore = _get_fut_stats(X, self.window_width)
        shift_expanded, scale_expanded = _expand_params(X, self.var_str, self.shift, self.scale)

        fut_mean_corrected, fut_std_corrected = _correct_fut_stats(fut_mean, fut_std, self.var_str, shift_expanded, scale_expanded)
        fut_corrected = (fut_zscore * fut_std_corrected) + fut_mean_corrected
        return fut_corrected
  

def _reshape(ds, window_width):
    """
    Helper function for `fit` that splits the year and day
    dimensions of the time-coordinate and bookends the years
    e.g. (Dec15:31 + whole year + Jan1:15) if window_width is 31 days.
    ---------------------
    Parameters:
    ds : Xarray Dataset
    window_width: the size of the rolling window.

    Returns:
    ds_rsh : Reshaped Xarray dataset
    """
    if 'time' not in ds.coords and 'index' in ds.coords:
        ds = ds.rename({'index': 'time'})
    assert 'time' in ds.coords

    split = lambda g: (g.rename({'time': 'day'})
                       .assign_coords(day=g.time.dt.dayofyear.values))
    ds_split = ds.groupby('time.year').apply(split)
    
    early_Jans = ds_split.isel(day = slice(None, window_width//2))
    late_Decs = ds_split.isel(day = slice(-window_width//2,None))
    
    ds_rsh = xr.concat([late_Decs,ds_split,early_Jans],dim='day')
    return ds_rsh


def _calc_stats(df, window_width):
    """
    Helper function for `fit` that calculates the rolling mean and  
    standard deviation for each day of the year across all years.
    ---------------------
    Parameters:
    df : Pandas dataframe
    window_width: the size of the rolling window.

    Returns:
    df_mean : Means for each day of year across all years
    df_std:  Standard deviations for each day of year across all years
    """

    ds = df.to_xarray()
    ds_rsh = _reshape(ds, window_width)
    
    ds_rolled = ds_rsh.rolling(day=window_width, center=True).construct('win_day')
    
    n = window_width//2+1
    ds_mean = ds_rolled.mean(dim=['year','win_day']).isel(day=slice(n,-n))
    ds_std = ds_rolled.std(dim=['year','win_day']).isel(day=slice(n,-n))

    df_mean = ds_mean.to_dataframe()
    df_std = ds_std.to_dataframe()
    return df_mean, df_std
                     

def _get_params(hist_mean, hist_std, meas_mean, meas_std):   
    """
    Helper function for `fit` that calculates the shift and scale parameters
    for z-score correction by comparing the historical and measured
    daily means and standard deviations.
    ---------------------
    Parameters:
    hist_mean : Mean calculated using the moving window 
        for each day on an average year from the historical
        model
    hist_std : Standard deviation calculated using the 
        moving window for each day on an average year
        from the historical model
    meas_mean : Mean calculated using the moving window 
        for each day on an average year from the measurements
    meas_std : Standard deviation calculated using the 
        moving window for each day on an average year
        from the measurements

    Returns:
    shift : The value by which to adjust the future mean.
    scale : The value by which to adjust the future standard deviation.
    """ 
    shift = meas_mean - hist_mean
    scale = meas_std / hist_std
    return shift, scale
                     

# Helpers for Predict
def _expand_params(df, var_str, shift, scale):
    """
    Helper function for `predict` that expands the shift and scale parameters
    from a 365-day average year, to the length of the future dataframe.
    ---------------------
    Parameters:
    df : Pandas dataframe
    var_str :  The key associated with the target dataframe variable
    shift : The value by which to adjust the future mean.
    scale : The value by which to adjust the future standard deviation.
    
    Returns:
    shift_expanded : The value by which to adjust the future mean, 
        repeated over the length of the dataframe.
    scale_expanded : The value by which to adjust the future standard deviation,
        repeated over the length of the dataframe.
    """ 
    repeats = (df.shape[0]/shift[var_str].shape[0])
    shift_expanded = np.repeat(shift[var_str], repeats)
    scale_expanded = np.repeat(scale[var_str], repeats)
    return shift_expanded, scale_expanded


def _get_fut_stats (df, window_width):
    """
    Helper function for `predict` that calculates statistics
    for the future dataset
    ---------------------
    Parameters:
    df : Pandas dataframe
    window_width: the size of the rolling window.
    
    Returns:
    fut_mean : Mean calculated using the moving window 
        for each day of the future model
    fut_std : Standard deviation calculated using the 
        moving window for each day of the future model
    fut_zscore: Z-Score coefficient calculated by comparing
        the dataframe values, the means, and standared
        deviations.
    """ 
    fut_mean = df.rolling(window_width, center=True).mean()
    fut_std = df.rolling(window_width, center=True).std()
    fut_zscore = (df - fut_mean) / fut_std
    return fut_mean, fut_std, fut_zscore


def _correct_fut_stats(fut_mean, fut_std, var_str, shift_expanded, scale_expanded):
    """
    Helper function for `predict` that adjusts future statistics 
    by shift and scale parameters.
    ---------------------
    Parameters:
    fut_mean : Mean calculated using the moving window 
        for each day of the future model
    fut_std : Standard deviation calculated using the 
        moving window for each day of the future model
    fut_zscore: Z-Score coefficient calculated by comparing
        the dataframe values, the means, and standared
        deviations.
    shift_expanded : The value by which to adjust the future mean, 
        repeated over the length of the dataframe.
    scale_expanded : The value by which to adjust the future standard deviation,
        repeated over the length of the dataframe.
    Returns:
    
    """ 
    fut_mean_corrected = fut_mean[var_str] + shift_expanded
    fut_std_corrected = fut_std[var_str] * scale_expanded
    return fut_mean_corrected, fut_std_corrected
