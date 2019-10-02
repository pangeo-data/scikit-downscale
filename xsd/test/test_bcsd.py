import numpy as np
import pandas as pd
import xarray as xr

import xsd

np.random.seed(3)


def make_random_data(loc=10, sigma=1, bounds=(0, -90, 360, 90),
                     size=(240, 10, 20), name=None, dist='normal'):

    if dist == 'normal':
        data = np.random.normal(loc, sigma, size)
    elif dist == 'lognormal':
        data = np.random.lognormal(loc, sigma, size)
    coords = {'time': pd.date_range('1970-01', freq='MS', periods=size[0]),
              'lat': np.linspace(bounds[1], bounds[3], size[1], endpoint=True),
              'lon': np.linspace(bounds[0], bounds[2], size[2])}
    return xr.DataArray(data, dims=('time', 'lat', 'lon'), coords=coords,
                        name=name)


def test_bcsd_random_temperature():

    obs = make_random_data(loc=10, sigma=1, bounds=(-125, 24, -66, 50),
                           size=(240, 104, 236), name='obs')
    train = make_random_data(loc=8, sigma=1.4, bounds=(-140, 20, -50, 60),
                             size=(240, 40, 80), name='train')
    pred = make_random_data(loc=11, sigma=1.5, bounds=(-140, 20, -50, 60),
                            size=(360, 40, 80), name='pred')

    out = xsd.bcsd.bcsd(obs, train, pred, var='temperature')

    assert out.shape[0] == 360
    assert out.shape[1] == 104
    assert out.shape[2] == 236
    assert out.mean() > 2.5  # shift is positive  8-->11
    assert out.mean() < 3.5  # shift is positive  8-->11


def test_bcsd_random_precip():

    obs = make_random_data(loc=0.05, sigma=0.01, bounds=(-125, 24, -66, 50),
                           size=(240, 104, 236), name='obs', dist='lognormal')
    train = make_random_data(loc=0.1, sigma=0.05, bounds=(-140, 20, -50, 60),
                             size=(240, 40, 80), name='train',
                             dist='lognormal')
    pred = make_random_data(loc=0.08, sigma=0.04, bounds=(-140, 20, -50, 60),
                            size=(360, 40, 80), name='pred', dist='lognormal')

    out = xsd.bcsd.bcsd(obs, train, pred, var='pr')
    out.to_netcdf('test.nc')

    assert out.shape[0] == 360
    assert out.shape[1] == 104
    assert out.shape[2] == 236
    # anoms should be around 0.8 for this test
    assert out.mean() < 1
    assert out.mean() > 0.6, print(out)
