import pytest
import pandas as pd
import xarray as xr
import collections
import numpy as np


@pytest.fixture
def tas_series():
    def _tas_series(values, start="7/1/2000"):
        coords = collections.OrderedDict()
        for dim, n in zip(("time", "lon", "lat"), values.shape):
            if dim == "time":
                coords[dim] = pd.date_range(start, periods=n, freq=pd.DateOffset(days=1))
            else:
                coords[dim] = xr.IndexVariable(dim, np.arange(n))

        return xr.DataArray(
            values,
            coords=coords,
            dims=list(coords.keys()),
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": "K",
            },
        )

    return _tas_series


@pytest.fixture
def qds_month():
    dims = ("quantile", "month")
    source = xr.Variable(dims=dims,
                         data=np.zeros((5, 12)))
    target = xr.Variable(dims=dims,
                         data=np.ones((5, 12))*2)

    return xr.Dataset(data_vars={"source": source,
                                 "target": target},
                      coords={"quantile": [0, .3, 5., 7, 1],
                              "month": range(1, 13)},
                      attrs={"group": "time.month",
                             "window": 1})

