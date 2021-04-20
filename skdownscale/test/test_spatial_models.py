import numpy as np
import pandas as pd
import pytest
import xarray as xr

from skdownscale.spatial_models import SpatialDisaggregator


@pytest.fixture
def test_dataset(request):
    n = 3650
    x = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, n)) * 0.5
    start_time = "1950-01-01"
    time = xr.cftime_range(
        start=start_time, freq="D", periods=len(x), calendar="standard"
    )
    ds = xr.Dataset(
        {request.param: (["time", "lon", "lat"], x[:, np.newaxis, np.newaxis])},
        coords={
            "index": time,
            "time": time,
            "lon": (["lon"], [1.0]),
            "lat": (["lat"], [1.0]),
        },
    )
    return ds


@pytest.mark.parametrize(
    "test_dataset, var_name",
    [
        pytest.param("temperature", "temperature"),
        pytest.param("precipitation", "precipitation"),
    ],
    indirect=["test_dataset"],
)
def test_spatialdisaggregator_fit(test_dataset, var_name):
    time = pd.date_range(start="2017-01-01", end="2020-01-01")
    data_X = np.linspace(0, 1, len(time))

    X = xr.DataArray(data_X, name=var_name, dims=["time"], coords={"time": time})
    # groupby_type = X.time.dt.dayofyear
    climo = test_dataset[var_name].groupby("time.dayofyear").mean()

    spatial_disaggregator = SpatialDisaggregator(var_name)
    scale_factor = spatial_disaggregator.fit(test_dataset, climo, var_name)

    if var_name == "temperature":
        np.testing.assert_allclose(
            scale_factor, test_dataset[var_name].groupby("time.dayofyear") - climo
        )
    elif var_name == "precipitation":
        np.testing.assert_allclose(
            scale_factor, test_dataset[var_name].groupby("time.dayofyear") / climo
        )


@pytest.mark.parametrize(
    "test_dataset, var_name",
    [
        pytest.param("temperature", "temperature"),
        pytest.param("precipitation", "precipitation"),
    ],
    indirect=["test_dataset"],
)
def test_spatialdisaggregator_predict(test_dataset, var_name):
    time = pd.date_range(start="2017-01-01", end="2020-01-01")
    data_X = np.linspace(0, 1, len(time))

    scale_factor = xr.DataArray(
        data_X, name=var_name, dims=["time"], coords={"time": time}
    )
    # groupby_type = scale_factor.time.dt.dayofyear
    climo = test_dataset[var_name].groupby("time.dayofyear").mean()

    spatial_disaggregator = SpatialDisaggregator(var_name)
    data_downscaled = spatial_disaggregator.predict(test_dataset, climo, var_name)

    if var_name == "temperature":
        np.testing.assert_allclose(
            data_downscaled, test_dataset[var_name].groupby("time.dayofyear") + climo
        )
    elif var_name == "precipitation":
        np.testing.assert_allclose(
            data_downscaled, test_dataset[var_name].groupby("time.dayofyear") * climo
        )
