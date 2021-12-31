import numpy as np
import pytest
import xarray as xr
from sklearn.preprocessing import StandardScaler

from skdownscale.pointwise_models import AnalogRegression, PointWiseDownscaler, QuantileMapper

from . import make_linear_reg_pipeline, random_grid_data, random_point_data

dask = pytest.importorskip('dask')


@pytest.mark.parametrize(
    ('X', 'y'),
    [
        # with numpy arrays
        (random_point_data(n_points=3, n_vars=3), random_point_data(n_points=3, n_vars=1)['a'],),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'],
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'],
        ),
        # with dask arrays
        (
            random_point_data(n_points=3, n_vars=3).chunk({'point': 1}),
            random_point_data(n_points=3, n_vars=1)['a'].chunk({'point': 1}),
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'].chunk({'y': 1, 'x': 1}),
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'].chunk({'y': 1, 'x': 1}),
        ),
    ],
)
def test_pointwise_model(X, y):
    pipeline = make_linear_reg_pipeline()
    model = PointWiseDownscaler(model=pipeline)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred.values  # otherwise some of the code will not be tested when input is chunked
    assert isinstance(y_pred, type(y))
    assert y_pred.sizes == y.sizes
    assert y.chunks == y_pred.chunks

    model = PointWiseDownscaler(model=AnalogRegression(thresh=0))
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred.values  # otherwise some of the code will not be tested when input is chunked
    assert isinstance(y_pred, type(y))
    assert y_pred.sizes['variable'] == model._model.n_outputs
    for dim in y.sizes:
        assert y_pred.sizes[dim] == y.sizes[dim]
    for dim in y.chunksizes:
        assert y_pred.chunksizes[dim] == y.chunksizes[dim]


@pytest.mark.parametrize(
    'X',
    [
        # with numpy arrays
        random_point_data(n_points=3, n_vars=3),
        random_point_data(n_points=3, n_vars=1),
        random_grid_data(grid_shape=(2, 3), n_vars=3),
        random_grid_data(grid_shape=(2, 3), n_vars=3),
        # with dask arrays
        random_point_data(n_points=3, n_vars=3).chunk({'point': 1}),
        random_point_data(n_points=3, n_vars=1).chunk({'point': 1}),
        random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
        random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
    ],
)
def test_pointwise_model_transform(X):
    scaler = StandardScaler()
    model = PointWiseDownscaler(model=scaler)
    model.fit(X)
    x_trans = model.transform(X)
    # Q: should transform return the same type it recieved? Or should we always return a DataArray
    x_trans_ds = x_trans.to_dataset('variable')
    assert isinstance(x_trans, xr.DataArray)
    assert x_trans_ds.sizes == X.sizes
    assert X.chunks == x_trans_ds.chunks


@pytest.mark.parametrize(
    ('X', 'y'),
    [
        # with numpy arrays
        (random_point_data(n_points=3, n_vars=3), random_point_data(n_points=3, n_vars=1)['a'],),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'],
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'],
        ),
        # with dask arrays
        (
            random_point_data(n_points=3, n_vars=3).chunk({'point': 1}),
            random_point_data(n_points=3, n_vars=1)['a'].chunk({'point': 1}),
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'].chunk({'y': 1, 'x': 1}),
        ),
        (
            random_grid_data(grid_shape=(2, 3), n_vars=3).chunk({'y': 1, 'x': 1}),
            random_grid_data(grid_shape=(2, 3), n_vars=1)['a'].chunk({'y': 1, 'x': 1}),
        ),
    ],
)
def test_pointwise_model_attributes(X, y):
    scaler = StandardScaler()
    model = PointWiseDownscaler(model=scaler)
    model.fit(X)

    # testing an attribute of a single object, such that the output size should be the same as the model size
    key = 'n_features_in_'
    dtype = 'int64'
    attrs = model.get_attr(key, dtype)
    assert isinstance(attrs, xr.DataArray)
    assert attrs.sizes == model._models.sizes
    assert attrs.dtype == dtype

    # testing an attribute of an arbitrary length, providing a template output for shape
    key = 'scale_'
    dtype = 'float64'
    template_output = X.isel(time=0).drop('time')
    template_output = template_output.expand_dims({'var': np.arange(3)})
    attrs = model.get_attr(key, dtype=dtype, template_output=template_output['a'])
    assert isinstance(attrs, xr.DataArray)
    assert attrs.sizes == template_output.sizes
    assert attrs.dtype == dtype
