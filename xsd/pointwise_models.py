import numpy as np
import xarray as xr

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression


def fit_model(X, y, model=None):
    # reshape input data
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # return a len 1 scalar of dtype np.object
    # there is likely a better way to do this
    out = np.empty((1), dtype=np.object)
    out[:] = [model.fit(X, y)]
    out = out.squeeze()
    print(out.shape)
    return out


def predict(model, X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # pull item() out because model is a np.scalar
    out = model.item().predict(X).squeeze()
    print(out.shape)
    return out


class PointWiseDownscaler:
    def __init__(self, model, dim='time'):
        self._dim = dim
        self._model = model
        self._fit_models = None

    def fit(self, X, y):
        """Fit the model"""
        self._fit_models = xr.apply_ufunc(fit_model, X, y,
                                          kwargs=dict(model=self._model),
                                          vectorize=True,
                                          dask='parallelized',
                                          output_dtypes=[np.object],
                                          input_core_dims=[[self._dim], [self._dim]])

    def predict(self, X, *predict_params):
        """Apply transforms to the data, and predict with the final estimator"""       
        return xr.apply_ufunc(predict, models, X,
                              vectorize=True,
                              dask='parallelized',
                              output_dtypes=[X.dtype],
                              input_core_dims=[[], [self._dim]],
                              output_core_dims=[[self._dim]])


# End of day summary
# - downscaler seems to work when dask chunks are full x/y space
# - but currently failing on subgrid chunks. This seems like a bug in dask or xarray
# - next steps are to 1) write tests, 2) implement a few simple GARD like models, 3) put together an example notebook