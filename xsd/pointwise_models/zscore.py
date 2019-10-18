from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
import pandas as pd
import numpy as np

class ZScoreRegressor(LinearModel, RegressorMixin):
    """ Z Score Regressor bias correction model wrapper

    Apply a scikit-learn model (e.g. Pipeline) point-by-point. The pipeline
    must implement the fit and predict methods.
    """

    def __init__(self, n_analogs=200, kdtree_kwargs={}, query_kwargs={}, lr_kwargs={}):

        self.n_analogs = n_analogs
        self.kdtree_kwargs = kdtree_kwargs
        self.query_kwargs = query_kwargs
        self.lr_kwargs = lr_kwargs

        
        
    
    def fit(self, X, y, window_width):
        """ Fit Z-Score Model
             

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
        
        X_mean, X_std, X_zscore = _get_circ_stats(X, window_width)
        y_mean, y_std, y_zscore = _get_circ_stats(y, window_width)
        
        shift, scale = _calc_correction_params(X_mean, X_std, y_mean, y_std)

        self.shift = shift
        self.scale = scale
        
        return self    
        
        
    def predict(self, X, window_width=31, var_str='U'):
        """ Predict
        
        This is where I change my future (X input is future raw data) based on fit
        
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
        corrected : pd.DataFrame, shape (n_samples, 1)
            Returns corrected values.
        """
        
        corrected = _z_score_correction(X, var_str, self.shift, self.scale, window_width)

        
        return corrected
  


#Helpers
# Helpers for Fit
def _hack_circular_rolling(df):
    df_avyear = df.groupby(lambda x: x.dayofyear).mean()

    last = df_avyear.iloc[-15:]
    first = df_avyear.iloc[:15]

    df_circ = pd.concat([last, df_avyear, first], ignore_index=True)
    return df_avyear, df_circ;

        
def _calc_circ_stats(df_avyear, df_circ, window_width): 
    df_mean = df_circ.rolling(window_width, center=True).mean()
    df_circ_mean = df_mean.iloc[15:380]

    df_std = df_circ.rolling(window_width, center=True).std()
    df_circ_std = df_std.iloc[15:380]

    df_circ_zscore = (df_avyear - df_mean) / df_std
    return df_circ_mean, df_circ_std, df_circ_zscore;

        
def _get_circ_stats(df, window_width):
    df_av, df_circ = _hack_circular_rolling(df)
    df_circ_mean, df_circ_std, df_circ_zscore = _calc_circ_stats(df_av, df_circ, window_width)
    return df_circ_mean, df_circ_std, df_circ_zscore
                     

def _calc_correction_params(hist_mean, hist_std, meas_mean, meas_std):    
    shift = meas_mean - hist_mean
    scale = meas_std / hist_std
    return shift, scale
                     

# Helpers for Predict
def _expand_params(df, var_str, shift, scale):
    repeats = (df.shape[0]/shift[var_str].shape[0])
    shift_expanded = np.repeat(shift[var_str], repeats)
    scale_expanded = np.repeat(scale[var_str], repeats)
    return shift_expanded, scale_expanded


def _get_fut_stats (df, window_width):
    ds_mean = df.rolling(window_width, center=True).mean()
    ds_std = df.rolling(window_width, center=True).std()
    ds_zscore = (df - df_mean) / df_std
    return ds_zscore


def _correct_fut_stats(fut_mean, fut_std, var_str, shift_expanded, scale_expanded):
    fut_mean_corrected = fut_mean[var_str] + shift_expanded
    fut_std_corrected = fut_std[var_str] * scale_expanded
    return fut_mean_corrected, fut_std_corrected


def _z_score_correction(ds_fut, var_str, shift, scale, window_width):
    fut_mean, fut_std, fut_zscore = _get_fut_stats(ds_fut, window_width)

    shift_expanded, scale_expanded = _expand_params(fut, var_str, shift, scale)

    fut_mean_corrected, fut_std_corrected = _correct_fut_stats(fut_mean, fut_std, var_str, shift_expanded, scale_expanded)
    fut_corrected = (fut_zscore * fut_std_corrected) + fut_mean_corrected
    return fut_corrected