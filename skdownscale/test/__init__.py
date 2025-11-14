import importlib
import string
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError('Minimum version not satisfied')
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f'requires {modname}')
    return has, func


has_dask, requires_dask = _importorskip('dask')

if has_dask:
    import dask

    dask.config.set(scheduler='single-threaded')


def random_point_data(n_points=1, n_times=100, n_vars=1):
    ds = xr.Dataset()
    size = (n_times, n_points)
    dims = ('time', 'point')
    times = pd.date_range('1979-01-01', freq='1D', periods=n_times)
    for vname in string.ascii_lowercase[:n_vars]:
        ds[vname] = xr.DataArray(np.random.random(size=size), dims=(dims), coords={'time': times})
    return ds


def random_grid_data(grid_shape=(2, 3), n_times=100, n_vars=1):
    ds = xr.Dataset()
    size = (n_times,) + grid_shape
    dims = ('time', 'y', 'x')
    times = pd.date_range('1979-01-01', freq='1D', periods=n_times)
    for vname in string.ascii_lowercase[:n_vars]:
        ds[vname] = xr.DataArray(np.random.random(size=size), dims=(dims), coords={'time': times})
    return ds


def make_linear_reg_pipeline():
    steps = []

    steps.append(('standardize', StandardScaler()))
    steps.append(('linear regression', LinearRegression()))

    return Pipeline(steps)
