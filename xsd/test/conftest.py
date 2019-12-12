import pytest
import pandas as pd
import xarray as xr



@pytest.fixture
def tas_series():
    def _tas_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": "K",
            },
        )

    return _tas_series
