from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
import pandas as pd
import numpy as np

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
        
        X_mean, X_std, X_zscore = _get_circ_stats(X, self.window_width)
        y_mean, y_std, y_zscore = _get_circ_stats(y, self.window_width)
        
        shift, scale = _calc_correction_params(X_mean, X_std, y_mean, y_std)

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
        fut_mean, fut_std, fut_zscore = _get_fut_stats(X, selfwindow_width)
        shift_expanded, scale_expanded = _expand_params(S, self.var_str, shift, scale)

        fut_mean_corrected, fut_std_corrected = _correct_fut_stats(fut_mean, fut_std, var_str, shift_expanded, scale_expanded)
        fut_corrected = (fut_zscore * fut_std_corrected) + fut_mean_corrected
        return fut_corrected
  


#Helpers
# Helpers for Fit
def _hack_circular_rolling(df, window_width):
    """
    Helper function for `fit` that creates a bookended average year 
    e.g. (Dec15:31 + whole year + Jan1:15) if window_width is 31 days.
    This is to get average values for each day across all years, 
    and so that the rolling function wraps around 
    (Jan 1st rolling stats are informed by late December)
    ---------------------
    Parameters:
    df : Pandas dataframe
    window_width: the size of the rolling window.

    Returns:
    df_avyear : Average values for each day across all years
    df_circ: df_avyear with bookends
    """
    df_avyear = df.groupby(lambda x: x.dayofyear).mean()

    bookend = int(window_width-1)/2
    last = df_avyear.iloc[-bookend:]
    first = df_avyear.iloc[:bookend]

    df_circ = pd.concat([last, df_avyear, first], ignore_index=True)
    return df_avyear, df_circ;

        
def _calc_circ_stats(df_avyear, df_circ, window_width): 
    """
    Helper function for `fit` that calculates statistics (mean, std, and zscore) 
    on a bookended average year for each day using a moving window.
    ---------------------
    Parameters:
    df_avyear : Average values for each day across all years
    df_circ: df_avyear with bookends
    window_width: the size of the rolling window.

    Returns:
    circ_mean : Mean calculated using the moving window 
        for each day on an average year
    circ_std : Standard deviation calculated using the 
        moving window for each day on an average year
    """
    bookend = int(window_width-1)/2
    df_mean = df_circ.rolling(window_width, center=True).mean()
    circ_mean = df_mean.iloc[bookend:365+bookend]

    df_std = df_circ.rolling(window_width, center=True).std()
    circ_std = df_std.iloc[bookend:365+bookend]
    return circ_mean, circ_std;

        
def _get_circ_stats(df, window_width):
    """
    Helper function for `fit` that generates bookended average
    year and calculates statistics (mean, std, and zscore) 
    on that bookended average year for each day using a moving window.
    ---------------------
    Parameters:
    df : Pandas dataframe
    window_width: the size of the rolling window.

    Returns:
    circ_mean : Mean calculated using the moving window 
        for each day on an average year
    circ_std : Standard deviation calculated using the 
        moving window for each day on an average year
    """
    df_av, df_circ = _hack_circular_rolling(df, window_width)
    circ_mean, circ_std = _calc_circ_stats(df_av, df_circ, window_width)
    return circ_mean, circ_std
                     

def _calc_correction_params(hist_mean, hist_std, meas_mean, meas_std):   
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
