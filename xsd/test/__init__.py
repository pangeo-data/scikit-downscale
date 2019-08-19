import string

import numpy as np
import pandas as pd
import xarray as xr


def random_point_data(n_points=1, n_times=100, n_vars=1):
    ds = xr.Dataset()
    size = (n_times, n_points)
    dims = ('time', 'point')
    times = pd.date_range('1979-01-01', freq='1D', periods=n_times)
    for vname in string.ascii_lowercase[:n_vars]:
        ds[vname] = xr.DataArray(np.random.random(
            size=size), dims=(dims), coords={'time': times})
    return ds


def random_grid_data(grid_shape=(2, 3), n_times=100, n_vars=1):
    ds = xr.Dataset()
    size = (n_times, ) + grid_shape
    dims = ('time', 'y', 'x')
    times = pd.date_range('1979-01-01', freq='1D', periods=n_times)
    for vname in string.ascii_lowercase[:n_vars]:
        ds[vname] = xr.DataArray(np.random.random(
            size=size), dims=(dims), coords={'time': times})
    return ds
