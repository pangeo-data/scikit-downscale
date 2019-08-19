from sklearn.linear_model import LinearRegression
import xarray as xr

from . import random_point_data, random_grid_data

from xsd.pointwise_models import PointWiseDownscaler


def test_pointwise_1d():
    y = random_point_data(n_points=3, n_vars=1)
    X = random_point_data(n_points=3, n_vars=3)
    model = PointWiseDownscaler(model=LinearRegression())
    model.fit(X, y, )
    y_pred = model.predict(X)
    assert isinstance(y_pred, xr.DataArray)
    assert y_pred.shape == y.shape


def test_pointwise_2d():
    y = random_grid_data(n_points=3, n_vars=1)
    X = random_grid_data(n_points=3, n_vars=3)
    model = PointWiseDownscaler(model=LinearRegression())
    model.fit(X, y, )
    y_pred = model.predict(X)
    assert isinstance(y_pred, xr.DataArray)
    assert y_pred.shape == y.shape
